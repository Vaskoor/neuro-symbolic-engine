# SPDX-License-Identifier: MIT
#
# neurosym.core.types
# -------------------
# Shared data contract for the Neuro-Symbolic Reasoning Engine.
#
# Every subsystem (knowledge graph, Datalog reasoner, SHACL validator,
# neural backend, LLM backend and the orchestrating engine) speaks the
# language defined in this module. Keeping the contract in one place makes
# the system fully decoupled yet interoperable.
#
# Python 3.11+ / pydantic v2.

"""Canonical data types exchanged between all Neuro-Symbolic Engine subsystems."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------
class QueryType(str, Enum):
    """Enumeration of supported query modalities.

    The orchestrating engine uses :class:`QueryType` to route a request
    through the most appropriate backend, while still allowing the
    cascading fallback (KG -> symbolic -> neural -> LLM) for any modality.
    """

    SPARQL = "SPARQL"
    DATALOG = "DATALOG"
    OWL = "OWL"
    NEURAL = "NEURAL"
    LLM = "LLM"
    HYBRID = "HYBRID"


# ---------------------------------------------------------------------------
# Triple
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    """An RDF triple whose fields are plain strings.

    Subjects and predicates are always URI strings (absolute IRIs or
    prefixed CURIEs such as ``ex:authoredBy``). Objects may be URI strings
    or lexical literal strings (e.g. ``"true"``, ``"2024"``). Using strings
    keeps the contract serialization-agnostic and lets the symbolic layer
    avoid hard dependencies on rdflib term types.
    """

    model_config = ConfigDict(frozen=True)

    s: str = Field(description="Subject — a URI string (absolute IRI or CURIE).")
    p: str = Field(description="Predicate — a URI string (absolute IRI or CURIE).")
    o: str = Field(description="Object — a URI string or a lexical literal string.")

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.s, self.p, self.o)

    def key(self) -> tuple[str, str, str]:
        """Stable hashable key used by the DatalogEngine fact table."""
        return self.as_tuple()


# ---------------------------------------------------------------------------
# Rule (Datalog)
# ---------------------------------------------------------------------------
class Rule(BaseModel):
    """A Horn-style Datalog rule: ``head :- body_1, body_2, ...``.

    Body and head patterns are dicts with ``s``/``p``/``o`` keys. Any value
    starting with ``?`` (e.g. ``?x``) is a variable; everything else is a
    constant that must match the corresponding triple field exactly.

    Example::

        Rule(
            name="derive_has_institution",
            body=[
                {"s": "?x", "p": "ex:authoredBy", "o": "?y"},
                {"s": "?y", "p": "ex:affiliatedWith", "o": "?z"},
            ],
            head={"s": "?x", "p": "ex:hasInstitution", "o": "?z"},
            priority=90,
            description="A paper has the institution of its authors.",
        )
    """

    name: str = Field(description="Globally-unique rule name; used in proof traces.")
    body: list[dict[str, str]] = Field(
        default_factory=list,
        description="Conjunctive body: list of {s,p,o} triple patterns with ?vars.",
    )
    head: dict[str, str] = Field(
        description="Single {s,p,o} pattern (with ?vars) instantiated on match.",
    )
    priority: int = Field(
        default=0,
        ge=0,
        description="Higher priority rules fire first within a fixpoint iteration.",
    )
    description: str = Field(default="", description="Human-readable rationale.")


# ---------------------------------------------------------------------------
# Proof trace
# ---------------------------------------------------------------------------
class ProofStep(BaseModel):
    """A single node in a recursive proof tree.

    A *leaf* fact has ``rule_name == "fact"`` (or ``"kg:lookup"``) and an
    empty ``premises`` list. A *derived* fact records the rule that derived
    it and the :class:`ProofStep` s justifying each body pattern.
    """

    rule_name: str = Field(description="Name of the rule that produced this step.")
    premises: list[ProofStep] = Field(
        default_factory=list,
        description="Justifying sub-proofs (recursive).",
    )
    conclusion: Triple = Field(description="The triple concluded by this step.")

    def depth(self) -> int:
        """Maximum depth of the proof subtree rooted at this step."""
        if not self.premises:
            return 1
        return 1 + max((p.depth() for p in self.premises), default=0)

    def flatten(self) -> list[ProofStep]:
        """Pre-order traversal of the proof tree (useful for rendering)."""
        out: list[ProofStep] = [self]
        for p in self.premises:
            out.extend(p.flatten())
        return out


# Resolve the self-referential ``list[ProofStep]`` annotation now that the
# class body is fully defined.
ProofStep.model_rebuild()


class ProofTrace(BaseModel):
    """A complete proof justifying a :class:`ReasoningResult`."""

    steps: list[ProofStep] = Field(
        default_factory=list,
        description="Root proof steps (typically one).",
    )
    final_conclusion: Triple = Field(description="The final triple concluded.")
    depth: int = Field(default=0, ge=0, description="Max recursion depth of the tree.")

    def as_flat_steps(self) -> list[ProofStep]:
        out: list[ProofStep] = []
        for s in self.steps:
            out.extend(s.flatten())
        return out


# ---------------------------------------------------------------------------
# SHACL violation
# ---------------------------------------------------------------------------
class Violation(BaseModel):
    """A single SHACL validation failure.

    Mirrors a subset of the SHACL ``ValidationReport`` structure so that
    callers can render results without depending on a SHACL library.
    """

    focus_node: str = Field(description="The node that violated a constraint.")
    path: Optional[str] = Field(default=None, description="Property path that failed (sh:path).")
    value: Optional[str] = Field(default=None, description="Offending value (sh:value).")
    constraint: str = Field(description="Constraint component, e.g. 'sh:minCount', 'sh:datatype'.")
    message: str = Field(description="Human-readable explanation.")
    severity: str = Field(default="sh:Violation", description="sh:Violation/Warning/Info.")
    shape: Optional[str] = Field(default=None, description="Shape that was evaluated.")
    source_shape: Optional[str] = Field(
        default=None, description="Property/node shape IRI that owns the constraint."
    )


# ---------------------------------------------------------------------------
# Reasoning result
# ---------------------------------------------------------------------------
SourceLabel = Literal["kg", "symbolic", "neural", "llm", "hybrid"]


class ReasoningResult(BaseModel):
    """The unified answer object returned by :class:`NeuroSymbolicEngine.query`.

    The ``answer`` field is deliberately polymorphic:

    * KG / symbolic queries return ``list[dict[str, str]]`` (variable bindings
      or {s,p,o} rows).
    * Neural / LLM backends return free-form ``str`` text.

    ``confidence`` is in ``[0.0, 1.0]``; the engine assigns deterministic
    defaults per source (KG=1.0, symbolic=0.9, neural configurable, llm=0.5).
    """

    query: str = Field(description="The original query text.")
    answer: Optional[list[dict[str, str]] | str] = Field(
        default=None, description="Structured rows or free-form text."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Engine confidence in the answer."
    )
    proof: Optional[ProofTrace] = Field(
        default=None, description="Justifying proof tree (if available)."
    )
    source: SourceLabel = Field(description="Which backend produced the answer.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock latency.")
    notes: str = Field(default="", description="Diagnostics / caveats.")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class Query(BaseModel):
    """An incoming reasoning request."""

    text: str = Field(description="Query text (SPARQL string, Datalog pattern text, ...).")
    type: QueryType = Field(default=QueryType.DATALOG, description="Routing hint.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured parameters. For DATALOG-style queries use "
            "``{'pattern': {'s': '?x', 'p': 'ex:authoredBy', 'o': '?y'}}``. "
            "For SPARQL leave ``text`` as the query and ``params`` empty."
        ),
    )


__all__ = [
    "QueryType",
    "Triple",
    "Rule",
    "ProofStep",
    "ProofTrace",
    "Violation",
    "ReasoningResult",
    "Query",
    "SourceLabel",
]
