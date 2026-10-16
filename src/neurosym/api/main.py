"""
FastAPI service for the Neuro-Symbolic Reasoning Engine.

Endpoints
---------
    POST /query     — hybrid cascade: KG → symbolic → neural → LLM fallback.
    POST /assert    — assert an RDF triple into the knowledge graph.
    GET  /sparql    — run a SPARQL 1.1 query against the graph.
    POST /reason    — pure symbolic reasoning with proof trace.
    GET  /explain   — retrieve a stored proof trace by `trace_id`.
    GET  /validate  — run SHACL validation.
    GET  /health    — liveness + capacity report.

Design notes
------------
* The FastAPI module imports cleanly even if the core/llm stack is not yet
  wired — the engine is constructed lazily on the first request via
  `_engine()` so `import neurosym.api.main` always succeeds.
* CORS is enabled with an env-configurable allow-list.
* Structured request/response logging is wired as ASGI middleware and emits
  a per-request `X-Trace-Id` header.
* `trace_id` → proof-trace lookups are stored in-process in `app.state.trace_store`
  (swap for Redis / Postgres in production).

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# --- FastAPI + pydantic ------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# --- Logging -----------------------------------------------------------------
LOG_LEVEL = os.environ.get("NEUROSYM_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":%(message)r}',
)
logger = logging.getLogger("neurosym.api")
logger.setLevel(LOG_LEVEL)


# =============================================================================
# Request / response models
# =============================================================================


class QueryRequest(BaseModel):
    """Body for `POST /query`."""

    text: str = Field(..., description="Natural-language or formal query.")
    type: str | None = Field(
        default=None,
        description=(
            "Optional query-type hint (one of SPARQL, DATALOG, OWL, NEURAL, "
            "LLM, HYBRID). Defaults to HYBRID."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional override of engine parameters (e.g. pattern dict).",
    )


class AssertRequest(BaseModel):
    """Body for `POST /assert`."""

    s: str = Field(..., description="Subject (URI or prefixed name).")
    p: str = Field(..., description="Predicate (URI or prefixed name).")
    o: str = Field(..., description="Object (URI, prefixed name, or literal).")


class ReasonRequest(BaseModel):
    """Body for `POST /reason`."""

    query: str = Field(..., description="Symbolic query (variable or boolean).")


class HealthResponse(BaseModel):
    """Response for `GET /health`."""

    status: str
    kg_triples: int
    rules: int
    embeddings_trained: bool


# =============================================================================
# Engine wiring (lazy)
# =============================================================================


def _build_engine() -> Any:
    """Construct the NeuroSymbolicEngine with deterministic, env-driven config.

    Imports are lazy so the FastAPI module imports cleanly even when the full
    core/llm stack is not yet wired (e.g. during CI smoke tests or static
    analysis).

    Honours these env vars:
      * NEUROSYM_ONTOLOGY   — path to the OWL ontology (.ttl)
      * NEUROSYM_RULES      — path to the Datalog rules (.yaml)
      * NEUROSYM_LLM_ENABLED — '1' to enable LLM fallback (default off)
      * NEUROSYM_OFFLINE    — '1' to disable network (default off)
    """
    # Imported here on purpose — keeps `import neurosym.api.main` cheap.
    from neurosym.core.config import DatalogConfig, EngineConfig, KGConfig  # type: ignore
    from neurosym.core.engine import NeuroSymbolicEngine  # type: ignore

    ontology = os.environ.get("NEUROSYM_ONTOLOGY", "config/ontology.ttl")
    rules_path = os.environ.get("NEUROSYM_RULES", "config/rules.yaml")
    ttl_files = [ontology] if Path(ontology).exists() else []
    rules_files = [rules_path] if Path(rules_path).exists() else []
    cfg = EngineConfig(
        kg=KGConfig(turtle_files=ttl_files),
        datalog=DatalogConfig(rules_files=rules_files),
    )
    return NeuroSymbolicEngine(config=cfg)


def _serialise(value: Any) -> Any:
    """Best-effort pydantic-v2 / dataclass / list / dict serialisation.

    Walks containers recursively so pydantic models nested in lists / dicts
    are also converted to plain JSON-safe values.
    """
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return _serialise(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _serialise(value.dict())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if isinstance(value, tuple):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    # Last resort: try vars() (dataclasses, plain objects).
    try:
        return {str(k): _serialise(v) for k, v in vars(value).items()}
    except TypeError:
        return str(value)


def _build_query(req: QueryRequest, *, default_type: str = "HYBRID") -> Any:
    """Convert an HTTP request body into a `neurosym.core.types.Query`.

    Honours optional `req.type` (one of SPARQL/DATALOG/OWL/NEURAL/LLM/HYBRID)
    and passes through `req.params` verbatim — when `params` carries a
    `pattern` dict (e.g. `{"s": "?x", "p": "ex:authoredBy", "o": "?y"}`),
    the engine's KG/symbolic stages can use it directly.

    If `req.params` is empty and the request looks like a Datalog pattern
    (e.g. `"?x ex:authoredBy ?y"`), use `QueryEngine.build_query()` to parse
    the text into a structured `pattern` + `select` and inject them into
    `params` so the engine's symbolic stage can answer.
    """
    from neurosym.core.types import Query, QueryType  # type: ignore

    qtype_str = (req.type or default_type).upper()
    try:
        qtype = QueryType(qtype_str)
    except ValueError:
        qtype = QueryType.HYBRID

    params = dict(req.params)
    # If the caller didn't pre-supply a `pattern`/`patterns` and the text
    # looks like a Datalog triple pattern (or NL), parse it on their behalf.
    is_sparql = qtype == QueryType.SPARQL or req.text.strip().lower().startswith(
        ("select", "ask", "construct", "describe")
    )
    if not is_sparql and "pattern" not in params and "patterns" not in params:
        try:
            from neurosym.reasoning.queries import QueryEngine  # type: ignore

            parsed = QueryEngine().build_query(req.text, query_type=qtype)
            # Merge any parsed pattern/patterns/select into our params.
            for k, v in parsed.params.items():
                if k not in params:
                    params[k] = v
        except Exception as exc:  # noqa: BLE001 — fall back to plain Query
            logger.debug("QueryEngine parsing failed (%s); passing text raw", exc)
    return Query(text=req.text, type=qtype, params=params)


# =============================================================================
# Application factory
# =============================================================================


def create_app() -> FastAPI:
    """Build and return a configured FastAPI application instance."""

    app = FastAPI(
        title="Neuro-Symbolic Reasoning Engine",
        version="0.1.0",
        description=(
            "Combining OWL, SHACL, Datalog, GNNs, and LLMs for Explainable Knowledge Inference."
        ),
        license_info={"name": "MIT"},
    )

    # --- CORS ----------------------------------------------------------------
    cors_origins = [
        o.strip() for o in os.environ.get("NEUROSYM_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- In-process state (swap for Redis/Postgres in production) ------------
    app.state.engine = None
    app.state.trace_store: dict[str, Any] = {}

    # --- Request/response logging middleware --------------------------------
    @app.middleware("http")
    async def _log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        logger.info(
            "req id=%s method=%s path=%s",
            rid,
            request.method,
            request.url.path,
        )
        try:
            response: Response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 — middleware must never raise
            ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "req id=%s status=500 latency_ms=%.2f err=%r",
                rid,
                ms,
                exc,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error", "trace_id": rid},
                headers={"X-Trace-Id": rid},
            )
        ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "req id=%s status=%s latency_ms=%.2f",
            rid,
            response.status_code,
            ms,
        )
        response.headers["X-Trace-Id"] = rid
        return response

    # --- Lazy engine accessor ------------------------------------------------
    def _engine() -> Any:
        if app.state.engine is None:
            try:
                app.state.engine = _build_engine()
            except ImportError as exc:
                raise HTTPException(
                    status_code=501,
                    detail=(
                        "neurosym core engine not available — install or wire "
                        f"the core layer. Missing import: {exc.name or exc}"
                    ),
                ) from exc
        return app.state.engine

    # =========================================================================
    # Routes
    # =========================================================================

    @app.get("/", tags=["meta"], summary="Service root")
    def root() -> dict[str, str]:
        return {"name": "Neuro-Symbolic Reasoning Engine", "version": "0.1.0"}

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Liveness + capacity probe",
    )
    def health() -> HealthResponse:
        """Return service status, KG triple count, rule count, and whether
        neural embeddings have been trained.
        """
        try:
            eng = _engine()
            kg_triples = 0
            rules = 0
            embeddings_trained = False
            # Triple count via engine introspection methods (preferred) or KG.
            if hasattr(eng, "triple_count"):
                kg_triples = int(eng.triple_count())
            elif getattr(eng, "kg", None) is not None and hasattr(eng.kg, "count"):
                kg_triples = int(eng.kg.count())
            # Rule count from the Datalog engine.
            datalog = getattr(eng, "datalog", None)
            if datalog is not None:
                rules_list = getattr(datalog, "rules", None)
                if isinstance(rules_list, list):
                    rules = len(rules_list)
                elif hasattr(datalog, "list_rules"):
                    rules = len(datalog.list_rules())
            # Embeddings trained iff a neural backend is wired.
            neural = getattr(eng, "neural", None)
            embeddings_trained = neural is not None
            return HealthResponse(
                status="ok",
                kg_triples=kg_triples,
                rules=rules,
                embeddings_trained=embeddings_trained,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — health probes must not 500
            logger.warning("health probe degraded: %s", exc)
            return HealthResponse(
                status="degraded",
                kg_triples=0,
                rules=0,
                embeddings_trained=False,
            )

    @app.post(
        "/query",
        tags=["reasoning"],
        summary="Hybrid cascade query (KG → symbolic → neural → LLM fallback)",
    )
    async def query(req: QueryRequest) -> dict[str, Any]:
        eng = _engine()
        try:
            q = _build_query(req, default_type="HYBRID")
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"neurosym.core.types not available: {exc}",
            )
        try:
            result = await eng.query(q)  # async cascade
        except AttributeError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"engine.query() unavailable: {exc}",
            )
        serialised = _serialise(result)
        trace_id = (
            serialised.get("trace_id")
            or str(serialised.get("source", "")) + "-" + uuid.uuid4().hex[:8]
        )
        serialised["trace_id"] = trace_id
        app.state.trace_store[trace_id] = serialised.get("proof")
        return serialised

    @app.post(
        "/assert",
        tags=["knowledge"],
        summary="Assert an RDF triple into the knowledge graph",
    )
    def assert_triple(req: AssertRequest) -> dict[str, Any]:
        eng = _engine()
        try:
            eng.assert_triple(req.s, req.p, req.o)
        except AttributeError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"engine.assert_triple() unavailable: {exc}",
            )
        return {"status": "ok", "triple": [req.s, req.p, req.o]}

    @app.get(
        "/sparql",
        tags=["knowledge"],
        summary="Run a SPARQL 1.1 query against the knowledge graph",
    )
    def sparql(
        q: str = Query(..., description="SPARQL 1.1 query string."),
    ) -> dict[str, Any]:
        eng = _engine()
        kg = getattr(eng, "kg", None)
        if kg is None:
            raise HTTPException(
                status_code=501,
                detail="engine has no knowledge graph attached",
            )
        # Prefer the canonical name `sparql_query`; fall back to `query_sparql`.
        if hasattr(kg, "sparql_query"):
            rows: Any = kg.sparql_query(q)
        elif hasattr(kg, "query_sparql"):
            rows = kg.query_sparql(q)
        else:
            raise HTTPException(
                status_code=501,
                detail="knowledge graph exposes no SPARQL accessor",
            )
        if not isinstance(rows, list):
            rows = list(rows) if hasattr(rows, "__iter__") else []
        return {"query": q, "results": rows, "count": len(rows)}

    @app.post(
        "/reason",
        tags=["reasoning"],
        summary="Pure symbolic reasoning with proof trace",
    )
    async def reason(req: ReasonRequest) -> dict[str, Any]:
        eng = _engine()
        try:
            q = _build_query(
                QueryRequest(text=req.query, type="DATALOG"),
                default_type="DATALOG",
            )
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"neurosym.core.types not available: {exc}",
            )
        try:
            result = await eng.query(q)  # cascade prefers symbolic for DATALOG
        except AttributeError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"engine.query() unavailable: {exc}",
            )
        serialised = _serialise(result)
        trace_id = (
            serialised.get("trace_id")
            or str(serialised.get("source", "")) + "-" + uuid.uuid4().hex[:8]
        )
        serialised["trace_id"] = trace_id
        app.state.trace_store[trace_id] = serialised.get("proof")
        return serialised

    @app.get(
        "/explain",
        tags=["reasoning"],
        summary="Retrieve a stored proof trace by `trace_id`",
    )
    def explain(
        trace_id: str = Query(..., description="Proof trace identifier."),
    ) -> dict[str, Any]:
        if trace_id not in app.state.trace_store:
            raise HTTPException(
                status_code=404,
                detail=f"trace_id {trace_id!r} not found",
            )
        return {"trace_id": trace_id, "proof": app.state.trace_store[trace_id]}

    @app.get(
        "/validate",
        tags=["knowledge"],
        summary="Run SHACL validation on the knowledge graph",
    )
    def validate() -> dict[str, Any]:
        eng = _engine()
        try:
            report: Any = eng.validate_shacl()
        except AttributeError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"engine.validate_shacl() unavailable: {exc}",
            )
        violations = _serialise(report)
        conforms = not violations
        return {
            "conforms": conforms,
            "violations": violations if isinstance(violations, list) else [],
            "count": len(violations) if isinstance(violations, list) else 0,
        }

    # Expose the lazy accessor on the app for tests.
    app.state._engine = _engine  # type: ignore[attr-defined]
    return app


# =============================================================================
# Module-level app instance (used by `uvicorn neurosym.api.main:app`)
# =============================================================================

app: FastAPI = create_app()


# =============================================================================
# `python -m neurosym.api.main` entry point (dev convenience)
# =============================================================================


def _main() -> int:
    """Run uvicorn with this module's app (dev mode)."""
    import uvicorn  # local import — keeps the module import-light.

    host = os.environ.get("NEUROSYM_HOST", "0.0.0.0")
    port = int(os.environ.get("NEUROSYM_PORT", "8000"))
    uvicorn.run(
        "neurosym.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
