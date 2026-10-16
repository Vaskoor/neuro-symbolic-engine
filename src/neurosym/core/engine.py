# SPDX-License-Identifier: MIT
#
# neurosym.core.engine
# ---------------------
# The orchestrating :class:`NeuroSymbolicEngine`.
#
# The engine wires together four reasoning backends and exposes a single
# async ``query`` entry point. Each request is routed through a configurable
# *cascade* — by default ``kg -> symbolic -> neural -> llm`` — and the first
# backend that can answer wins. This makes the system robust to missing or
# unconfigured backends (neural / LLM are optional and guarded).
#
# Layering rationale
# ~~~~~~~~~~~~~~~~~~~
# * **KG** answers from asserted facts only (highest confidence).
# * **Symbolic** runs Datalog forward-chaining over KG + derived facts and
#   answers from inferred triples (still high confidence, fully auditable).
# * **Neural** delegates to an injected backend (e.g. an embedding-based
#   retriever or a classifier) — optional.
# * **LLM** is the last-resort free-form reasoner — optional.
#
# Every successful result carries a :class:`ProofTrace` (KG lookup leaf, or a
# recursive Datalog derivation tree) so that ``explain(result)`` can always
# reconstruct why the engine believes its answer.

"""The orchestrating neuro-symbolic engine with cascading backends."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..knowledge.datalog import DatalogEngine
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.shacl import SHACLValidator
from .config import EngineConfig, load_config
from .types import (
    ProofStep,
    ProofTrace,
    Query,
    QueryType,
    ReasoningResult,
    Triple,
    Violation,
)


# ---------------------------------------------------------------------------
# Optional backend protocols (structural typing — no concrete deps needed).
# ---------------------------------------------------------------------------
@runtime_checkable
class NeuralBackend(Protocol):
    async def infer(self, query: Query) -> ReasoningResult:  # pragma: no cover
        ...


@runtime_checkable
class LLMBackend(Protocol):
    async def complete(self, query: Query) -> ReasoningResult:  # pragma: no cover
        ...


class NeuroSymbolicEngine:
    """Orchestrates the KG, symbolic reasoner and (optional) neural/LLM backends."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        *,
        kg: Optional[KnowledgeGraph] = None,
        datalog: Optional[DatalogEngine] = None,
        neural: Optional[NeuralBackend] = None,
        llm: Optional[LLMBackend] = None,
    ) -> None:
        self.config: EngineConfig = config or EngineConfig()
        self.kg: KnowledgeGraph = kg or KnowledgeGraph(base_namespace=self.config.kg.base_namespace)
        self.datalog: DatalogEngine = datalog or DatalogEngine(seed=self.config.datalog.seed)
        self.neural: Optional[NeuralBackend] = neural
        self.llm: Optional[LLMBackend] = llm
        self._shacl: Optional[SHACLValidator] = None
        self._shacl_initialized: bool = False
        self._bootstrap()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path) -> "NeuroSymbolicEngine":
        return cls(config=load_config(path))

    def _bootstrap(self) -> None:
        """Load configured Turtle files, rules, and seed the Datalog fact table."""
        for ttl in self.config.kg.turtle_files:
            self.kg.load_turtle(ttl)
        for rules_file in self.config.datalog.rules_files:
            self.datalog.load_rules(rules_file)
        self.sync_facts_from_kg()

    # ------------------------------------------------------------------
    # Fact / triple management
    # ------------------------------------------------------------------
    def assert_triple(self, s: str, p: str, o: str) -> Triple:
        """Assert a triple into the KG *and* the Datalog fact table."""
        triple = Triple(s=s, p=p, o=o)
        self.kg.add_triple(s, p, o)
        self.datalog.add_fact(triple)
        return triple

    def sync_facts_from_kg(self) -> int:
        """Re-seed the Datalog base facts from the KG (clears derived facts).

        Returns the number of base facts loaded.
        """
        triples = [Triple(s=row["s"], p=row["p"], o=row["o"]) for row in self.kg.get_triples()]
        self.datalog.set_facts(triples)
        return len(triples)

    # ------------------------------------------------------------------
    # Querying (cascading backends)
    # ------------------------------------------------------------------
    async def query(self, q: Query) -> ReasoningResult:
        """Answer ``q`` via the configured cascade; first hit wins.

        The cascade is ``kg -> symbolic -> neural -> llm`` by default and is
        fully configurable via ``EngineConfig.cascade``.
        """
        t0 = time.perf_counter()
        for stage in self.config.cascade or ["kg", "symbolic", "neural", "llm"]:
            stage = stage.lower()
            result: Optional[ReasoningResult] = None
            if stage == "kg":
                result = self._try_kg(q)
            elif stage == "symbolic":
                result = self._try_symbolic(q)
            elif stage == "neural":
                result = await self._try_neural(q)
            elif stage == "llm":
                result = await self._try_llm(q)
            if result is not None:
                result.latency_ms = (time.perf_counter() - t0) * 1000.0
                return result
        # No backend could answer.
        return ReasoningResult(
            query=q.text,
            answer=None,
            confidence=0.0,
            proof=None,
            source="hybrid",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            notes="No backend in the cascade could answer this query.",
        )

    def _try_kg(self, q: Query) -> Optional[ReasoningResult]:
        """Direct lookup against asserted KG facts (no inference)."""
        if q.type == QueryType.SPARQL:
            rows = self.kg.sparql_query(q.text)
            if not rows:
                return None
            proof = self._leaf_proof(self._first_row_as_triple(rows))
            return ReasoningResult(
                query=q.text,
                answer=rows,
                confidence=1.0,
                proof=proof,
                source="kg",
                notes="SPARQL SELECT against asserted facts.",
            )
        pattern = q.params.get("pattern")
        if isinstance(pattern, dict):
            rows = self.kg.get_triples(
                s=pattern.get("s") or None,
                p=pattern.get("p") or None,
                o=pattern.get("o") or None,
            )
            if not rows:
                return None
            first = rows[0]
            conclusion = Triple(s=first["s"], p=first["p"], o=first["o"])
            return ReasoningResult(
                query=q.text,
                answer=rows,
                confidence=1.0,
                proof=self._leaf_proof(conclusion, rule_name="kg:lookup"),
                source="kg",
                notes="Direct triple lookup against asserted facts.",
            )
        return None

    def _try_symbolic(self, q: Query) -> Optional[ReasoningResult]:
        """Forward-chain Datalog rules and answer from inferred facts."""
        # Re-seed base facts (in case the KG grew since the last query) and
        # run a fresh fixpoint computation.
        self.sync_facts_from_kg()
        self.datalog.forward_chain(max_iterations=self.config.datalog.max_iterations)
        pattern = q.params.get("pattern")
        if not isinstance(pattern, dict):
            return None
        bindings = self.datalog.query(pattern)
        if not bindings:
            return None
        # Build a proof rooted at the first matching triple.
        first_binding = bindings[0]
        conclusion = self.datalog.query_conclusion(pattern, first_binding)
        proof = self._build_proof_trace(conclusion)
        return ReasoningResult(
            query=q.text,
            answer=bindings,
            confidence=0.9,
            proof=proof,
            source="symbolic",
            notes=f"Derived via Datalog forward-chaining ({len(bindings)} binding(s)).",
        )

    async def _try_neural(self, q: Query) -> Optional[ReasoningResult]:
        if self.neural is None:
            return None
        try:
            result = await self.neural.infer(q)
        except Exception as exc:  # pragma: no cover - defensive
            return ReasoningResult(
                query=q.text,
                answer=None,
                confidence=0.0,
                proof=None,
                source="neural",
                notes=f"Neural backend error: {exc}",
            )
        # Stamp the source in case the backend forgot.
        result.source = "neural"
        return result

    async def _try_llm(self, q: Query) -> Optional[ReasoningResult]:
        if self.llm is None:
            return None
        try:
            result = await self.llm.complete(q)
        except Exception as exc:  # pragma: no cover - defensive
            return ReasoningResult(
                query=q.text,
                answer=None,
                confidence=0.0,
                proof=None,
                source="llm",
                notes=f"LLM backend error: {exc}",
            )
        result.source = "llm"
        return result

    # ------------------------------------------------------------------
    # Proof reconstruction
    # ------------------------------------------------------------------
    def explain(self, result: ReasoningResult) -> ProofTrace:
        """Return the proof trace justifying ``result``.

        If the result already carries a proof (KG / symbolic / neural / LLM),
        it is returned verbatim; otherwise a trivial single-step trace is
        synthesized from the best-effort conclusion.
        """
        if result.proof is not None:
            return result.proof
        conclusion = self._answer_as_triple(result)
        step = ProofStep(
            rule_name=str(result.source),
            premises=[],
            conclusion=conclusion,
        )
        return ProofTrace(steps=[step], final_conclusion=conclusion, depth=1)

    def _build_proof_trace(self, conclusion: Triple) -> ProofTrace:
        step = self._build_proof_step(conclusion, set())
        return ProofTrace(
            steps=[step],
            final_conclusion=conclusion,
            depth=step.depth(),
        )

    def _build_proof_step(
        self,
        triple: Triple,
        _seen: set[tuple[str, str, str]],
    ) -> ProofStep:
        """Recursively reconstruct a proof tree from Datalog derivations.

        A guard set breaks cycles in (theoretically possible) cyclic
        derivations such as transitive closure rules.
        """
        key = triple.as_tuple()
        derivation = self.datalog.proof_for(triple)
        if derivation is None or key in _seen:
            # Base fact: asserted directly (no rule derived it).
            return ProofStep(rule_name="fact", premises=[], conclusion=triple)
        rule_name, premises = derivation
        seen = _seen | {key}
        sub_steps = [self._build_proof_step(p, seen) for p in premises]
        return ProofStep(
            rule_name=rule_name,
            premises=sub_steps,
            conclusion=triple,
        )

    @staticmethod
    def _leaf_proof(triple: Triple, rule_name: str = "kg:lookup") -> ProofTrace:
        step = ProofStep(rule_name=rule_name, premises=[], conclusion=triple)
        return ProofTrace(steps=[step], final_conclusion=triple, depth=1)

    def _first_row_as_triple(self, rows: list[dict[str, str]]) -> Triple:
        first = rows[0] if rows else {}
        return Triple(
            s=first.get("s", ""),
            p=first.get("p", ""),
            o=first.get("o", ""),
        )

    def _answer_as_triple(self, result: ReasoningResult) -> Triple:
        answer = result.answer
        if isinstance(answer, list) and answer and isinstance(answer[0], dict):
            row = answer[0]
            return Triple(
                s=row.get("s", ""),
                p=row.get("p", ""),
                o=row.get("o", ""),
            )
        if isinstance(answer, str) and answer:
            return Triple(s="?", p="llm:text", o=answer)
        return Triple(s="", p="", o="")

    # ------------------------------------------------------------------
    # SHACL validation
    # ------------------------------------------------------------------
    def validate_shacl(self) -> list[Violation]:
        """Validate the KG against configured SHACL shapes.

        Shapes are loaded from ``EngineConfig.shacl.shapes_files``; if none
        are provided, the validator looks for ``sh:NodeShape`` instances
        already present in the data graph.
        """
        if not self.config.shacl.enabled:
            return []
        if not self._shacl_initialized:
            shapes_kg: Optional[KnowledgeGraph] = None
            if self.config.shacl.shapes_files:
                shapes_kg = KnowledgeGraph(base_namespace=self.config.kg.base_namespace)
                for f in self.config.shacl.shapes_files:
                    shapes_kg.load_turtle(f)
            self._shacl = SHACLValidator(
                self.kg, shapes_kg if (shapes_kg and shapes_kg.count() > 0) else None
            )
            self._shacl_initialized = True
        assert self._shacl is not None
        return self._shacl.validate()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def triple_count(self) -> int:
        return self.kg.count()

    def derived_fact_count(self) -> int:
        return len(self.datalog.derivations)


__all__ = [
    "NeuroSymbolicEngine",
    "NeuralBackend",
    "LLMBackend",
]
