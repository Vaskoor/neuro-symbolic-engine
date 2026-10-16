# MIT License
#
# Copyright (c) 2024 Neuro-Symbolic Engine Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Proof construction for the symbolic reasoner.

The :class:`ProofBuilder` consumes the *rule-firing log* produced by the
:class:`neurosym.knowledge.datalog.DatalogEngine` (specifically its
``derivations`` map, which records for every derived triple the rule name
and the premise triples that fired it) and reconstructs a structured,
recursive :class:`~neurosym.core.types.ProofTrace` that justifies a goal
triple by walking the derivation graph backwards.

Each :class:`~neurosym.core.types.ProofStep` in the resulting tree carries:

* the name of the rule that fired (``"fact"`` for leaf KG facts),
* the recursive sub-proofs justifying each premise,
* the triple concluded by this step.

The firing-log input is normalised defensively — callers may pass any of:

* the DatalogEngine's ``derivations`` dict (mapping a triple key to a
  ``(rule_name, premises)`` tuple),
* a list of firing-log entries (each a ``dict`` / ``tuple`` / pydantic
  model exposing ``rule_name``, ``premises`` and ``conclusion``),
* a hybrid: a list whose entries are ``(conclusion_triple,
  (rule_name, premises))`` 2-tuples (so the caller can interleave assertions
  and derivations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

try:  # contract from sibling task ns-A
    from neurosym.core.types import (
        ProofStep,
        ProofTrace,
        Query,
        QueryType,
        ReasoningResult,
        Rule,
        SourceLabel,
        Triple,
        Violation,
    )

    _HAS_CONTRACT = True
except ImportError:  # pragma: no cover - fallback when ns-A not yet shipped
    _HAS_CONTRACT = False
    from enum import Enum
    from typing import Any, Literal, Optional

    from pydantic import BaseModel, ConfigDict, Field, field_validator

    class QueryType(str, Enum):  # type: ignore[no-redef]
        SPARQL = "SPARQL"
        DATALOG = "DATALOG"
        OWL = "OWL"
        NEURAL = "NEURAL"
        LLM = "LLM"
        HYBRID = "HYBRID"

    SourceLabel = Literal["kg", "symbolic", "neural", "llm", "hybrid"]

    class Triple(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback RDF triple — mirrors the real contract field names."""

        model_config = ConfigDict(frozen=True)

        s: str
        p: str
        o: str

        def as_tuple(self) -> tuple[str, str, str]:
            return (self.s, self.p, self.o)

        def key(self) -> tuple[str, str, str]:
            return self.as_tuple()

    class Rule(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback Datalog rule — mirrors the real contract."""

        name: str
        body: list[dict[str, str]] = Field(default_factory=list)
        head: dict[str, str]
        priority: int = Field(default=0, ge=0)
        description: str = ""

    class ProofStep(BaseModel):  # type: ignore[no-redef]
        """A single node in a recursive proof tree."""

        rule_name: str
        premises: list["ProofStep"] = Field(default_factory=list)
        conclusion: Triple

        @field_validator("conclusion", mode="before")
        @classmethod
        def _coerce_conclusion(cls, v: Any) -> Any:
            if v is None:
                return v
            return _coerce_triple(v)

        def depth(self) -> int:
            if not self.premises:
                return 1
            return 1 + max((p.depth() for p in self.premises), default=0)

        def flatten(self) -> list["ProofStep"]:
            out: list[ProofStep] = [self]
            for p in self.premises:
                out.extend(p.flatten())
            return out

    ProofStep.model_rebuild()

    class ProofTrace(BaseModel):  # type: ignore[no-redef]
        """A complete proof justifying a :class:`ReasoningResult`."""

        steps: list[ProofStep] = Field(default_factory=list)
        final_conclusion: Triple
        depth: int = Field(default=0, ge=0)

        @field_validator("final_conclusion", mode="before")
        @classmethod
        def _coerce_final(cls, v: Any) -> Any:
            if v is None:
                return v
            return _coerce_triple(v)

        @field_validator("steps", mode="before")
        @classmethod
        def _coerce_steps(cls, v: Any) -> Any:
            if v is None:
                return []
            if isinstance(v, (list, tuple, set)):
                out: list[ProofStep] = []
                for item in v:
                    if isinstance(item, ProofStep):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.append(ProofStep(**item))
                    else:
                        continue
                return out
            return [v] if v else []

        def as_flat_steps(self) -> list[ProofStep]:
            out: list[ProofStep] = []
            for s in self.steps:
                out.extend(s.flatten())
            return out

    class Violation(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback SHACL violation."""

        focus_node: str
        path: Optional[str] = None
        value: Optional[str] = None
        constraint: str
        message: str
        severity: str = "sh:Violation"
        shape: Optional[str] = None
        source_shape: Optional[str] = None

    class ReasoningResult(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback reasoning result."""

        query: str = ""
        answer: Optional[list[dict[str, str]] | str] = None
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)
        proof: Optional[ProofTrace] = None
        source: SourceLabel = "symbolic"  # type: ignore[assignment]
        latency_ms: float = Field(default=0.0, ge=0.0)
        notes: str = ""

    class Query(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback query."""

        text: str
        type: QueryType = QueryType.DATALOG  # type: ignore[assignment]
        params: dict[str, Any] = Field(default_factory=dict)


if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Coercion helpers — turn any triple-like value into a :class:`Triple`.
# ---------------------------------------------------------------------------
def _coerce_triple(value: Any) -> Triple:
    """Coerce a heterogeneous representation into a :class:`Triple`."""
    if value is None:
        raise ValueError("Cannot coerce None into a Triple")
    if isinstance(value, Triple):
        return value
    if isinstance(value, Mapping):
        if {"s", "p", "o"} <= set(value.keys()):
            return Triple(s=str(value["s"]), p=str(value["p"]), o=str(value["o"]))
        if {"subject", "predicate", "object"} <= set(value.keys()):
            return Triple(
                s=str(value["subject"]),
                p=str(value["predicate"]),
                o=str(value["object"]),
            )
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return Triple(s=str(value[0]), p=str(value[1]), o=str(value[2]))
    if hasattr(value, "s") and hasattr(value, "p") and hasattr(value, "o"):
        return Triple(
            s=str(getattr(value, "s")),
            p=str(getattr(value, "p")),
            o=str(getattr(value, "o")),
        )
    if hasattr(value, "subject") and hasattr(value, "predicate") and hasattr(value, "object"):
        return Triple(
            s=str(getattr(value, "subject")),
            p=str(getattr(value, "predicate")),
            o=str(getattr(value, "object")),
        )
    raise TypeError(f"Cannot coerce {type(value).__name__!r} into a Triple")


def _triple_key(value: Any) -> tuple[str, str, str]:
    """Return the (s, p, o) hashable key of any triple-like value."""
    return _coerce_triple(value).as_tuple()


def _coerce_triple_list(value: Any) -> list[Triple]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [_coerce_triple(value)]
    if isinstance(value, (list, tuple, set)):
        return [_coerce_triple(item) for item in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [_coerce_triple(item) for item in list(value)]
        except Exception:
            return [_coerce_triple(value)]
    return [_coerce_triple(value)]


def _triple_str(t: Triple | None) -> str:
    if t is None:
        return "(none)"
    return f"({t.s}, {t.p}, {t.o})"


# ---------------------------------------------------------------------------
# Firing-log normalisation
# ---------------------------------------------------------------------------
def _normalize_firing_log(
    firing_log: Any,
) -> dict[tuple[str, str, str], tuple[str, list[Triple]]]:
    """Normalise a firing log into ``{conclusion_key: (rule_name, premises)}``.

    Accepted shapes:

    * the DatalogEngine's ``derivations`` dict (mapping a triple key to a
      ``(rule_name, premises)`` tuple),
    * a list of dicts / pydantic models / tuples each describing one firing
      (with ``rule_name``, ``premises`` and ``conclusion`` keys/attrs),
    * a list of ``(conclusion, (rule_name, premises))`` 2-tuples.
    """
    derivations: dict[tuple[str, str, str], tuple[str, list[Triple]]] = {}

    if firing_log is None:
        return derivations

    # Shape 1: a mapping (derivations dict)
    if isinstance(firing_log, Mapping):
        for k, v in firing_log.items():
            try:
                key = _triple_key(k)
            except Exception:
                continue
            rule_name, premises = _extract_rule_and_premises(v)
            derivations[key] = (rule_name, premises)
        return derivations

    # Shape 2: a list of firing entries
    if isinstance(firing_log, Iterable) and not isinstance(firing_log, (str, bytes)):
        for entry in firing_log:
            rule_name, premises, conclusion = _extract_firing_entry(entry)
            if conclusion is None:
                continue
            derivations[_triple_key(conclusion)] = (rule_name, premises)
        return derivations

    return derivations


def _extract_rule_and_premises(value: Any) -> tuple[str, list[Triple]]:
    """Extract (rule_name, premises) from a derivations-dict value."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        rule_name = str(value[0])
        premises = _coerce_triple_list(value[1])
        return rule_name, premises
    if isinstance(value, Mapping):
        rule_name = str(
            value.get("rule_name")
            or value.get("rule")
            or value.get("name")
            or value.get("rule_id")
            or "rule"
        )
        premises = _coerce_triple_list(
            value.get("premises") or value.get("antecedents") or value.get("body") or []
        )
        return rule_name, premises
    if hasattr(value, "__len__") and len(value) >= 2:  # type: ignore[arg-type]
        return str(value[0]), _coerce_triple_list(value[1])
    return "rule", []


def _extract_firing_entry(entry: Any) -> tuple[str, list[Triple], Triple | None]:
    """Extract (rule_name, premises, conclusion) from a single firing entry."""
    if isinstance(entry, Mapping):
        rule_name = str(
            entry.get("rule_name")
            or entry.get("rule")
            or entry.get("name")
            or entry.get("rule_id")
            or "rule"
        )
        premises = _coerce_triple_list(
            entry.get("premises") or entry.get("antecedents") or entry.get("body") or []
        )
        conclusion = entry.get("conclusion") or entry.get("head") or entry.get("derived")
        conclusion_triple = _coerce_triple(conclusion) if conclusion is not None else None
        return rule_name, premises, conclusion_triple
    if isinstance(entry, (tuple, list)):
        # Possibilities:
        # (rule_name, premises, conclusion)
        # (rule_name, premises, conclusion, bindings)
        # (conclusion, (rule_name, premises))
        if len(entry) >= 3:
            rule_name = str(entry[0])
            premises = _coerce_triple_list(entry[1])
            conclusion = _coerce_triple(entry[2]) if entry[2] is not None else None
            return rule_name, premises, conclusion
        if len(entry) == 2:
            a, b = entry
            # Try (conclusion, (rule_name, premises)) first
            try:
                conc = _coerce_triple(a)
                rule_name, premises = _extract_rule_and_premises(b)
                return rule_name, premises, conc
            except Exception:
                pass
            # Otherwise (rule_name, premises) with no conclusion
            try:
                return str(a), _coerce_triple_list(b), None
            except Exception:
                return "rule", [], None
        return "rule", [], None
    # duck-typed object
    rule_name = (
        getattr(entry, "rule_name", None)
        or getattr(entry, "rule", None)
        or getattr(entry, "name", None)
        or "rule"
    )
    premises_val = (
        getattr(entry, "premises", None)
        or getattr(entry, "antecedents", None)
        or getattr(entry, "body", None)
        or []
    )
    conclusion_val = (
        getattr(entry, "conclusion", None)
        or getattr(entry, "head", None)
        or getattr(entry, "derived", None)
    )
    premises = _coerce_triple_list(premises_val)
    conclusion = None
    if conclusion_val is not None:
        try:
            conclusion = _coerce_triple(conclusion_val)
        except Exception:
            conclusion = None
    return str(rule_name), premises, conclusion


# ---------------------------------------------------------------------------
# ProofBuilder
# ---------------------------------------------------------------------------
class ProofBuilder:
    """Builds :class:`ProofTrace` objects from a DatalogEngine firing log.

    Example
    -------
    >>> builder = ProofBuilder()
    >>> derivations = {("b1", "ex:authoredBy", "Alice"):
    ...     ("author_of_inverse", [Triple(s="b1", p="ex:authorOf", o="Alice")])}
    >>> trace = builder.build_trace(derivations,
    ...     goal=Triple(s="b1", p="ex:authoredBy", o="Alice"))
    >>> print(format_proof(trace))
    """

    LEAF_RULE_NAME: str = "fact"

    def __init__(self, *, max_depth: int = 1024) -> None:
        self._max_depth = int(max_depth)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build_trace(
        self,
        firing_log: Any,
        *,
        goal: Any | None = None,
        base_facts: Iterable[Any] | None = None,
    ) -> ProofTrace:
        """Construct a recursive :class:`ProofTrace` for ``goal``.

        Parameters
        ----------
        firing_log:
            The DatalogEngine's ``derivations`` dict, or any of the
            alternative firing-log shapes accepted by
            :func:`_normalize_firing_log`.
        goal:
            The triple whose derivation the proof should justify. If
            ``None``, every entry in the firing log becomes a root step
            (useful for surfacing the whole derivation forest).
        base_facts:
            Optional iterable of triples known to be asserted (leaf) facts.
            Used only to label premises that are *not* in the firing log:
            if a premise is in ``base_facts`` it is labelled
            ``"fact"``; otherwise it is labelled ``"asserted"`` to flag
            that its provenance is unknown.

        Returns
        -------
        ProofTrace
            A trace whose ``steps`` is a singleton list (the root step for
            ``goal``) and whose ``final_conclusion`` is ``goal``.
        """
        derivations = _normalize_firing_log(firing_log)
        base_keys: set[tuple[str, str, str]] = set()
        if base_facts is not None:
            for bf in base_facts:
                try:
                    base_keys.add(_triple_key(bf))
                except Exception:
                    continue

        if goal is None:
            # Build a forest of all derived conclusions.
            steps: list[ProofStep] = []
            for key in derivations:
                step = self._build_step(key, derivations, base_keys, set(), 0)
                if step is not None:
                    steps.append(step)
            if not steps:
                # No derivations; return an empty-ish trace with a sentinel
                # final_conclusion (callers rarely hit this branch — they
                # always pass an explicit goal).
                empty = Triple(s="", p="", o="")
                return ProofTrace(steps=[], final_conclusion=empty, depth=0)
            final = steps[0].conclusion
            depth = max((s.depth() for s in steps), default=0)
            return ProofTrace(steps=steps, final_conclusion=final, depth=depth)

        goal_triple = _coerce_triple(goal)
        goal_key = goal_triple.as_tuple()
        root = self._build_step(goal_key, derivations, base_keys, set(), 0)
        if root is None:
            # Goal is not derived and not in base_facts — synthesise a leaf
            root = ProofStep(
                rule_name=self.LEAF_RULE_NAME,
                premises=[],
                conclusion=goal_triple,
            )
        return ProofTrace(steps=[root], final_conclusion=goal_triple, depth=root.depth())

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _build_step(
        self,
        key: tuple[str, str, str],
        derivations: dict[tuple[str, str, str], tuple[str, list[Triple]]],
        base_keys: set[tuple[str, str, str]],
        visited: set[tuple[str, str, str]],
        depth: int,
    ) -> ProofStep | None:
        """Recursively build a :class:`ProofStep` for the triple at ``key``."""
        if depth > self._max_depth:
            return ProofStep(
                rule_name="<depth-limit>",
                premises=[],
                conclusion=Triple(s=key[0], p=key[1], o=key[2]),
            )
        if key in visited:
            # Cycle in the derivation graph — return as a leaf to terminate.
            return ProofStep(
                rule_name="<cycle>",
                premises=[],
                conclusion=Triple(s=key[0], p=key[1], o=key[2]),
            )
        if key in derivations:
            rule_name, premises = derivations[key]
            visited2 = visited | {key}
            sub_steps: list[ProofStep] = []
            for prem in premises:
                prem_key = prem.as_tuple()
                sub = self._build_step(prem_key, derivations, base_keys, visited2, depth + 1)
                if sub is not None:
                    sub_steps.append(sub)
                else:
                    # Premise is a leaf fact — wrap as a fact step.
                    sub_steps.append(
                        ProofStep(
                            rule_name=self.LEAF_RULE_NAME,
                            premises=[],
                            conclusion=prem,
                        )
                    )
            return ProofStep(
                rule_name=rule_name,
                premises=sub_steps,
                conclusion=Triple(s=key[0], p=key[1], o=key[2]),
            )
        # Key not in derivations: leaf fact.
        rule_label = self.LEAF_RULE_NAME if key in base_keys else "asserted"
        return ProofStep(
            rule_name=rule_label,
            premises=[],
            conclusion=Triple(s=key[0], p=key[1], o=key[2]),
        )


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------
def format_proof(trace: ProofTrace) -> str:
    """Render a :class:`ProofTrace` as an indented, human-readable tree.

    The output begins with a ``GOAL:`` line stating the final conclusion
    and then walks the recursive proof tree in pre-order, indenting each
    sub-proof so the deduction chain is readable at a glance.
    """
    lines: list[str] = []
    final = getattr(trace, "final_conclusion", None)
    if final is not None:
        lines.append(f"GOAL: {_triple_str(final)}")
        lines.append("")
    if not trace.steps:
        lines.append("(no proof steps recorded)")
        return "\n".join(lines)
    counter = 0

    def render_step(step: ProofStep, indent: int, prefix: str) -> None:
        nonlocal counter
        counter += 1
        idx = counter
        pad = "  " * indent
        lines.append(f"{pad}{prefix}{idx}. rule = {step.rule_name}")
        lines.append(f"{pad}    conclusion = {_triple_str(step.conclusion)}")
        if step.premises:
            lines.append(f"{pad}    premises:")
            for i, sub in enumerate(step.premises, start=1):
                render_step(sub, indent + 2, f"{i}.")
        else:
            lines.append(f"{pad}    premises: (leaf)")

    for i, root in enumerate(trace.steps, start=1):
        render_step(root, 0, f"[{i}] sub-step ")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ProofBuilder",
    "format_proof",
]
