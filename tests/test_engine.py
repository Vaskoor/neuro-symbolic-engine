"""
Tests for `neurosym.core.engine.NeuroSymbolicEngine`.

End-to-end tests of the cascading backend pipeline:
    KG → symbolic → neural → LLM fallback.

Covers:
    1.  construction with default config
    2.  construction from a YAML config file
    3.  query() that hits the KG stage (source = "kg")
    4.  query() that hits the symbolic stage (source = "symbolic")
    5.  query() that hits the neural stage with a mock backend (source = "neural")
    6.  query() that hits the LLM stage with a mock backend (source = "llm")
    7.  query() that hits no backend (source = "hybrid", answer = None)
    8.  proof trace is present on KG/symbolic results
    9.  latency_ms is recorded and positive
    10. assert_triple() persists a triple and the next query sees it
    11. validate_shacl() returns a list of violations (possibly empty)
    12. explain(result) returns a ProofTrace

Runs offline — no network, no torch, no LLM. Tests use mock backends for the
neural and LLM stages so no external dependencies are required.

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""

from __future__ import annotations  # noqa: F401

import asyncio  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

# --- repo-local src layout bootstrap ---------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import pytest  # noqa: E402

import neurosym.core  # noqa: E402,F401  (correct import order)

# --- Module-under-test import; skip the whole file if unavailable -----------
try:
    from neurosym.core.config import DatalogConfig, EngineConfig, KGConfig  # type: ignore  # noqa: F401
    from neurosym.core.engine import NeuroSymbolicEngine  # type: ignore  # noqa: F401
    from neurosym.core.types import ProofTrace, Query, QueryType, ReasoningResult, Triple  # type: ignore  # noqa: F401

    HAS_ENG = True
    IMPORT_ERR: str | None = None
except ImportError as exc:  # pragma: no cover
    HAS_ENG = False
    IMPORT_ERR = str(exc)

pytestmark = pytest.mark.skipif(
    not HAS_ENG,
    reason=f"neurosym.core.engine not importable: {IMPORT_ERR}",
)


# =============================================================================
# Fixtures
# =============================================================================

REPO_ROOT = HERE.parent
BUNDLED_ONTOLOGY = REPO_ROOT / "config" / "ontology.ttl"
BUNDLED_RULES = REPO_ROOT / "config" / "rules.yaml"


def _make_config() -> EngineConfig:
    """Build an EngineConfig that wires the bundled ontology + rules."""
    ttl_files = [str(BUNDLED_ONTOLOGY)] if BUNDLED_ONTOLOGY.exists() else []
    rules_files = [str(BUNDLED_RULES)] if BUNDLED_RULES.exists() else []
    return EngineConfig(
        kg=KGConfig(turtle_files=ttl_files),
        datalog=DatalogConfig(rules_files=rules_files),
    )


@pytest.fixture
def engine() -> NeuroSymbolicEngine:
    """A NeuroSymbolicEngine with the bundled ontology + rules."""
    return NeuroSymbolicEngine(config=_make_config())


# --- Mock backends ----------------------------------------------------------


class _MockNeuralBackend:
    """Deterministic mock neural backend implementing the NeuralBackend protocol."""

    def __init__(self, answer_text: str = "neural-hypothesis") -> None:
        self.answer_text = answer_text
        self.calls: list[Query] = []

    async def infer(self, query: Query) -> ReasoningResult:
        self.calls.append(query)
        return ReasoningResult(
            query=query.text,
            answer=self.answer_text,
            confidence=0.7,
            source="neural",
            latency_ms=0.1,
            notes="mock-neural",
        )


class _MockLLMBackend:
    """Deterministic mock LLM backend implementing the LLMBackend protocol."""

    def __init__(self, answer_text: str = "llm-fallback-answer") -> None:
        self.answer_text = answer_text
        self.calls: list[Query] = []

    async def complete(self, query: Query) -> ReasoningResult:
        self.calls.append(query)
        return ReasoningResult(
            query=query.text,
            answer=self.answer_text,
            confidence=0.4,
            source="llm",
            latency_ms=0.1,
            notes="mock-llm",
        )


@pytest.fixture
def engine_with_neural() -> NeuroSymbolicEngine:
    """Engine with a mock neural backend."""
    return NeuroSymbolicEngine(
        config=_make_config(),
        neural=_MockNeuralBackend(),
    )


@pytest.fixture
def engine_with_llm() -> NeuroSymbolicEngine:
    """Engine with a mock LLM backend (and no neural backend)."""
    return NeuroSymbolicEngine(
        config=_make_config(),
        llm=_MockLLMBackend(),
    )


def _run(coro: Any) -> Any:
    """Run an async coroutine to completion under a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


# =============================================================================
# Construction tests
# =============================================================================


def test_construct_default() -> None:
    """1. NeuroSymbolicEngine() with no args constructs an empty engine."""
    eng = NeuroSymbolicEngine()
    assert eng is not None
    assert eng.kg is not None
    assert eng.datalog is not None
    assert eng.neural is None
    assert eng.llm is None


def test_construct_from_yaml(tmp_path: Path) -> None:
    """2. NeuroSymbolicEngine.from_config(path) loads a YAML config."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
name: test
version: 0.1.0
seed: 42
kg:
  base_namespace: http://example.org/sci#
datalog:
  seed: 42
  max_iterations: 50
""",
        encoding="utf-8",
    )
    eng = NeuroSymbolicEngine.from_config(str(cfg_path))
    assert eng.config.name == "test"
    assert eng.config.datalog.max_iterations == 50


# =============================================================================
# Cascade tests
# =============================================================================


def test_query_hits_kg(engine: NeuroSymbolicEngine) -> None:
    """3. A SPARQL query against asserted facts hits the KG stage."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    q = Query(
        text="SELECT ?s ?o WHERE { ?s rdfs:subClassOf ?o } LIMIT 5",
        type=QueryType.SPARQL,
    )
    result = _run(engine.query(q))
    assert result.source == "kg"
    assert result.confidence == 1.0
    assert isinstance(result.answer, list)
    assert len(result.answer) > 0


def test_query_hits_symbolic(engine: NeuroSymbolicEngine) -> None:
    """4. A Datalog query that requires forward-chaining hits the symbolic stage.

    Add instance data + verify the engine's symbolic stage (Datalog
    forward-chain) derives a `hasInstitution` triple via the bundled rule
    `derive_has_institution`.
    """
    if not BUNDLED_RULES.exists():  # pragma: no cover
        pytest.skip("config/rules.yaml not present")
    # Add the base facts the rule needs.
    engine.assert_triple("ex:P1", "ex:authoredBy", "ex:R1")
    engine.assert_triple("ex:R1", "ex:affiliatedWith", "ex:I1")
    # The KG stage will not match this pattern (variables) so it returns None;
    # the symbolic stage forward-chains and matches the derived fact.
    q = Query(
        text="?x ex:hasInstitution ?z",
        type=QueryType.DATALOG,
        params={"pattern": {"s": "?x", "p": "ex:hasInstitution", "o": "?z"}},
    )
    result = _run(engine.query(q))
    assert result.source == "symbolic"
    assert isinstance(result.answer, list)
    assert len(result.answer) > 0
    # Verify the binding contains ex:P1 / ex:I1.
    found = any(
        (b.get("x") == "ex:P1" or b.get("?x") == "ex:P1")
        and (b.get("z") == "ex:I1" or b.get("?z") == "ex:I1")
        for b in result.answer
        if isinstance(b, dict)
    )
    assert found, f"expected ex:P1/ex:I1 binding, got {result.answer}"


def test_query_hits_neural(engine_with_neural: NeuroSymbolicEngine) -> None:
    """5. A query with no KG/symbolic match falls through to the neural backend."""
    # A pattern that matches nothing in the KG or rules.
    q = Query(
        text="ex:NoSuch ex:NoSuchProp ?z",
        type=QueryType.DATALOG,
        params={"pattern": {"s": "ex:NoSuch", "p": "ex:NoSuchProp", "o": "?z"}},
    )
    result = _run(engine_with_neural.query(q))
    assert result.source == "neural"
    assert result.confidence == 0.7
    assert result.notes == "mock-neural"


def test_query_hits_llm(engine_with_llm: NeuroSymbolicEngine) -> None:
    """6. A query with no KG/symbolic match (and no neural backend) falls
    through to the LLM backend."""
    q = Query(
        text="ex:NoSuch ex:NoSuchProp ?z",
        type=QueryType.DATALOG,
        params={"pattern": {"s": "ex:NoSuch", "p": "ex:NoSuchProp", "o": "?z"}},
    )
    result = _run(engine_with_llm.query(q))
    assert result.source == "llm"
    assert result.confidence == 0.4
    assert result.notes == "mock-llm"


def test_query_no_backend_answers(engine: NeuroSymbolicEngine) -> None:
    """7. A query that no backend can answer returns source='hybrid' and None."""
    q = Query(
        text="ex:NoSuch ex:NoSuchProp ?z",
        type=QueryType.DATALOG,
        params={"pattern": {"s": "ex:NoSuch", "p": "ex:NoSuchProp", "o": "?z"}},
    )
    result = _run(engine.query(q))
    assert result.source == "hybrid"
    assert result.answer is None
    assert result.confidence == 0.0


# =============================================================================
# Proof + latency tests
# =============================================================================


def test_proof_trace_present_on_kg_result(engine: NeuroSymbolicEngine) -> None:
    """8. KG results carry a non-empty ProofTrace."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    q = Query(
        text="SELECT ?s ?o WHERE { ?s rdfs:subClassOf ?o } LIMIT 1",
        type=QueryType.SPARQL,
    )
    result = _run(engine.query(q))
    assert result.proof is not None
    # A ProofTrace has at least one step.
    proof: ProofTrace = result.proof
    assert isinstance(proof.steps, list)
    assert len(proof.steps) >= 1


def test_latency_recorded(engine: NeuroSymbolicEngine) -> None:
    """9. latency_ms is recorded and positive."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    q = Query(
        text="SELECT ?s WHERE { ?s a <http://www.w3.org/2002/07/owl#Class> } LIMIT 1",
        type=QueryType.SPARQL,
    )
    result = _run(engine.query(q))
    assert result.latency_ms > 0.0


def test_source_attribution(engine: NeuroSymbolicEngine) -> None:
    """10. The `source` field is always one of the canonical labels."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    valid_sources = {"kg", "symbolic", "neural", "llm", "hybrid"}
    q = Query(
        text="SELECT ?s WHERE { ?s a <http://www.w3.org/2002/07/owl#Class> } LIMIT 1",
        type=QueryType.SPARQL,
    )
    result = _run(engine.query(q))
    assert result.source in valid_sources


def test_assert_triple_persists(engine: NeuroSymbolicEngine) -> None:
    """11. assert_triple() adds a triple visible to subsequent KG queries."""
    engine.assert_triple("ex:TestS", "ex:TestP", "ex:TestO")
    q = Query(
        text="SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
        type=QueryType.SPARQL,
    )
    # The KG has lots of triples; filter for our test triple.
    result = _run(engine.query(q))
    rows = result.answer if isinstance(result.answer, list) else []
    found = any(
        isinstance(r, dict)
        and r.get("s") == "ex:TestS"
        and r.get("p") == "ex:TestP"
        and r.get("o") == "ex:TestO"
        for r in rows
    )
    assert found, f"asserted triple not visible in KG; got {len(rows)} rows"


def test_validate_shacl(engine: NeuroSymbolicEngine) -> None:
    """12. validate_shacl() returns a list of Violation objects."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    violations = engine.validate_shacl()
    assert isinstance(violations, list)
    # Each violation (if any) should have the canonical fields.
    for v in violations:
        assert hasattr(v, "focus_node")
        assert hasattr(v, "constraint")
        assert hasattr(v, "message")


def test_explain_returns_proof_trace(engine: NeuroSymbolicEngine) -> None:
    """13. explain(result) returns a ProofTrace for a KG result."""
    if not BUNDLED_ONTOLOGY.exists():  # pragma: no cover
        pytest.skip("config/ontology.ttl not present")
    q = Query(
        text="SELECT ?s ?o WHERE { ?s rdfs:subClassOf ?o } LIMIT 1",
        type=QueryType.SPARQL,
    )
    result = _run(engine.query(q))
    proof = engine.explain(result)
    assert proof is not None
    assert isinstance(proof, ProofTrace)
    assert len(proof.steps) >= 1
