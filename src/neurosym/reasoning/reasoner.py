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
"""Symbolic reasoner: Datalog forward-chaining + OWL subsumption + consistency.

The :class:`SymbolicReasoner` is the bridge between the deterministic
knowledge layer (:mod:`neurosym.knowledge`) and the rest of the
neuro-symbolic stack (the neural component and the LLM bridge).

Responsibilities (per the ``ns-B`` contract):

* :meth:`SymbolicReasoner.reason`     - forward-chain a
  :class:`~neurosym.knowledge.graph.KnowledgeGraph` against the
  registered Datalog rules, answer a :class:`~neurosym.core.types.Query`,
  and return a :class:`~neurosym.core.types.ReasoningResult` carrying the
  answer, the proof tree, and the derivation log.
* :meth:`SymbolicReasoner.check_consistency` - OWL disjointness-based
  contradiction detection (``X rdf:type A`` ∧ ``X rdf:type B`` ∧
  ``A owl:disjointWith B``).
* :meth:`SymbolicReasoner.subsumes`     - OWL subclass-closure subsumption
  (``class_a rdfs:subClassOf* class_b``).

The reasoner is **deterministic**: every fix-point iteration, BFS walk, and
internal fallback PRNG is seeded so two runs against the same
``(rules, kg, query)`` triple produce byte-identical results.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .proof import ProofBuilder, format_proof
from .queries import QueryEngine, is_variable

try:  # contract from sibling task ns-A
    from neurosym.core.types import (
        ProofStep,  # noqa: F401
        ProofTrace,  # noqa: F401
        Query,  # noqa: F401
        QueryType,  # noqa: F401
        ReasoningResult,  # noqa: F401
        Rule,  # noqa: F401
        SourceLabel,  # noqa: F401
        Triple,  # noqa: F401
    )
except ImportError:  # pragma: no cover - fallback when ns-A not yet shipped
    # Reuse the canonical fallback defined in :mod:`neurosym.reasoning.proof`
    # so every reasoning submodule shares structurally-compatible model
    # classes (a Triple built in queries.py is the same class as one built
    # in reasoner.py / proof.py).
    from .proof import (
        Query,
        QueryType,
        ReasoningResult,
        Rule,
        Triple,
    )

try:  # the DatalogEngine contract from sibling task ns-A
    from neurosym.knowledge.datalog import DatalogEngine as _ContractDatalogEngine

    _HAS_DATALOG = True
except Exception:  # pragma: no cover - fallback when ns-A not yet shipped
    _HAS_DATALOG = False
    _ContractDatalogEngine = None  # type: ignore[assignment, misc]

try:  # the KnowledgeGraph contract from sibling task ns-A
    from neurosym.knowledge.graph import KnowledgeGraph as _ContractKnowledgeGraph  # noqa: F401

    _HAS_KG = True
except Exception:  # pragma: no cover
    _HAS_KG = False


if TYPE_CHECKING:  # pragma: no cover
    from neurosym.knowledge.datalog import DatalogEngine
    from neurosym.knowledge.graph import KnowledgeGraph


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OWL/RDF predicate aliases — the engine accepts both the prefixed (CURIE)
# and full-IRI spellings so callers can use whichever is convenient.
# ---------------------------------------------------------------------------
_DISJOINT_PREDICATES: frozenset[str] = frozenset(
    {
        "owl:disjointWith",
        "http://www.w3.org/2002/07/owl#disjointWith",
    }
)
_TYPE_PREDICATES: frozenset[str] = frozenset(
    {
        "rdf:type",
        "a",  # SPARQL shorthand
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    }
)
_SUBCLASS_PREDICATES: frozenset[str] = frozenset(
    {
        "rdfs:subClassOf",
        "subClassOf",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    }
)


# ---------------------------------------------------------------------------
# KG adapter — accept whatever shape a KnowledgeGraph exposes.
# ---------------------------------------------------------------------------
def _coerce_triple(value: Any) -> Triple:
    """Coerce a heterogeneous value into a :class:`Triple`."""
    if value is None:
        raise ValueError("Cannot coerce None into a Triple")
    if isinstance(value, Triple):
        return value
    if isinstance(value, dict):
        if {"s", "p", "o"} <= set(value.keys()):
            return Triple(
                s=str(value["s"]),
                p=str(value["p"]),
                o=str(value["o"]),
            )
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


def _triple_key(t: Triple) -> tuple[str, str, str]:
    return t.as_tuple()


def _kg_iter_triples(kg: Any) -> list[Triple]:
    """Return all triples in ``kg`` as a list of :class:`Triple`.

    Tolerates the various accessor shapes a :class:`KnowledgeGraph` might
    expose: ``get_triples()`` (the canonical rdflib-backed method,
    returning ``list[dict[str, str]]``), ``triples`` (property/method
    returning Triples), ``all_triples()``, an ``__iter__`` yielding triples,
    or a ``query()`` wildcard call.
    """
    if kg is None:
        return []

    # 1. get_triples() — canonical API, returns list of dicts with s/p/o.
    if hasattr(kg, "get_triples"):
        try:
            res = kg.get_triples()
            return [_coerce_triple(t) for t in res]
        except TypeError:
            # Method may expect explicit (s, p, o) args; fall through.
            pass
        except Exception:
            pass

    # 2. .triples property / method returning Triple instances
    if hasattr(kg, "triples"):
        triples_attr = kg.triples
        if callable(triples_attr):
            try:
                triples_attr = triples_attr()
            except TypeError:
                triples_attr = []
        try:
            return [_coerce_triple(t) for t in triples_attr]
        except TypeError:
            pass

    # 3. .all_triples() method
    if hasattr(kg, "all_triples"):
        try:
            return [_coerce_triple(t) for t in kg.all_triples()]
        except TypeError:
            pass

    # 4. .get_facts() method (DatalogEngine-style)
    if hasattr(kg, "get_facts"):
        try:
            return [_coerce_triple(t) for t in kg.get_facts()]
        except TypeError:
            pass

    # 5. .query() wildcard
    if hasattr(kg, "query"):
        try:
            res = kg.query()
            return [_coerce_triple(t) for t in res]
        except TypeError:
            pass
        except Exception:
            pass

    # 6. direct iteration
    try:
        return [_coerce_triple(t) for t in kg]
    except TypeError:
        pass
    return []


def _kg_add_triple(kg: Any, s: str, p: str, o: str) -> None:
    """Add a triple to ``kg`` if the KG exposes a mutation method."""
    if kg is None:
        return
    # Prefer the canonical API (string s, p, o)
    if hasattr(kg, "add_triple"):
        try:
            kg.add_triple(s, p, o)
            return
        except TypeError:
            # add_triple takes a Triple
            try:
                kg.add_triple(Triple(s=s, p=p, o=o))
                return
            except Exception:
                pass
        except Exception:
            pass
    if hasattr(kg, "assert_triple"):
        try:
            kg.assert_triple(Triple(s=s, p=p, o=o))
            return
        except Exception:
            pass
    if hasattr(kg, "add_fact"):
        try:
            kg.add_fact(Triple(s=s, p=p, o=o))
            return
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pattern matching / substitution primitives (Datalog-semantics, dict patterns)
# ---------------------------------------------------------------------------
def _is_var(value: Optional[str]) -> bool:
    return value is not None and value.startswith("?")


def _match_pattern(
    pattern: dict[str, str],
    fact: Triple,
    bindings: dict[str, str],
) -> Optional[dict[str, str]]:
    """Match one triple pattern (dict with s/p/o keys) against one fact.

    Returns the (possibly extended) bindings dict, or ``None`` on failure.
    Variables (tokens starting with ``?``) are bound to the corresponding
    fact field; existing bindings must agree.
    """
    new_b = dict(bindings)
    for key in ("s", "p", "o"):
        if key not in pattern:
            continue
        val = pattern[key]
        fval = getattr(fact, key)
        if _is_var(val):
            if val in new_b:
                if new_b[val] != fval:
                    return None
            else:
                new_b[val] = fval
        else:
            if val != fval:
                return None
    return new_b


def _match_body(
    body: list[dict[str, str]],
    facts: list[Triple],
    bindings: dict[str, str],
    premises: list[Triple],
) -> list[tuple[dict[str, str], list[Triple]]]:
    """Recursively join all body patterns; return (bindings, premises) tuples."""
    if not body:
        return [(bindings, premises)]
    first, rest = body[0], body[1:]
    out: list[tuple[dict[str, str], list[Triple]]] = []
    for fact in facts:
        b = _match_pattern(first, fact, bindings)
        if b is not None:
            out.extend(_match_body(rest, facts, b, premises + [fact]))
    return out


def _subst(value: Optional[str], bindings: dict[str, str]) -> str:
    if value is None:
        return ""
    if _is_var(value):
        return bindings.get(value, value)
    return value


def _instantiate(head: dict[str, str], bindings: dict[str, str]) -> Triple:
    return Triple(
        s=_subst(head.get("s"), bindings),
        p=_subst(head.get("p"), bindings),
        o=_subst(head.get("o"), bindings),
    )


def _has_unbound_vars_in_triple(t: Triple) -> bool:
    return any(_is_var(v) for v in (t.s, t.p, t.o))


def _dedupe_premises(premises: list[Triple]) -> list[Triple]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Triple] = []
    for p in premises:
        k = p.as_tuple()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Internal fallback Datalog engine — used when the contract isn't importable
# ---------------------------------------------------------------------------
class _FallbackDatalogEngine:
    """A minimal, deterministic Datalog engine mirroring the contract API.

    Implements just enough of
    :class:`neurosym.knowledge.datalog.DatalogEngine` for the
    :class:`SymbolicReasoner` to forward-chain, query, and reconstruct proofs
    when the contract module is not yet importable.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self.rules: list[Rule] = []
        self._rules_by_name: dict[str, Rule] = {}
        self.facts: dict[tuple[str, str, str], Triple] = {}
        self.derivations: dict[tuple[str, str, str], tuple[str, list[Triple]]] = {}

    # ---- rule management ----
    def add_rule(self, rule: Rule) -> None:
        if rule.name in self._rules_by_name:
            self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
        self._rules_by_name[rule.name] = rule
        # Stable sort by descending priority (same as contract)
        self.rules.sort(key=lambda r: -getattr(r, "priority", 0))

    def list_rules(self) -> list[Rule]:
        return list(self.rules)

    # ---- fact management ----
    def set_facts(self, triples: Iterable[Triple]) -> None:
        self.facts = {}
        self.derivations = {}
        for t in triples:
            t = _coerce_triple(t)
            self.facts[t.as_tuple()] = t

    def add_fact(self, triple: Triple) -> bool:
        triple = _coerce_triple(triple)
        key = triple.as_tuple()
        if key in self.facts:
            return False
        self.facts[key] = triple
        return True

    def add_facts(self, triples: Iterable[Triple]) -> int:
        n = 0
        for t in triples:
            if self.add_fact(t):
                n += 1
        return n

    def get_facts(self) -> list[Triple]:
        return sorted(self.facts.values(), key=lambda t: t.as_tuple())

    def clear_derived(self) -> None:
        derived_keys = [k for k in self.facts if k in self.derivations]
        for k in derived_keys:
            del self.facts[k]
        self.derivations = {}

    # ---- forward chaining ----
    def forward_chain(self, max_iterations: int = 100) -> list[Triple]:
        newly_derived: list[Triple] = []
        for _iteration in range(max_iterations):
            facts_snapshot = self.get_facts()
            added_this_iteration = 0
            for rule in list(self.rules):
                body = list(getattr(rule, "body", []))
                head = getattr(rule, "head", None)
                if not head:
                    continue
                matches = _match_body(body, facts_snapshot, {}, [])
                seen_heads: set[tuple[str, str, str]] = set()
                for bindings, premises in matches:
                    head_triple = _instantiate(head, bindings)
                    key = head_triple.as_tuple()
                    if key in seen_heads:
                        continue
                    seen_heads.add(key)
                    if key in self.facts:
                        continue
                    if _has_unbound_vars_in_triple(head_triple):
                        # Unsafe rule (unbound var in head); skip but log.
                        continue
                    self.facts[key] = head_triple
                    self.derivations[key] = (rule.name, _dedupe_premises(premises))
                    newly_derived.append(head_triple)
                    added_this_iteration += 1
            if added_this_iteration == 0:
                break
        return newly_derived

    # ---- querying ----
    def query(self, pattern: dict[str, str]) -> list[dict[str, str]]:
        facts = self.get_facts()
        out: list[dict[str, str]] = []
        for fact in facts:
            b = _match_pattern(pattern, fact, {})
            if b is not None:
                out.append(dict(b))
        out.sort(key=lambda d: sorted(d.items()))
        return out

    def query_conclusion(self, pattern: dict[str, str], binding: dict[str, str]) -> Triple:
        return _instantiate(pattern, binding)

    def proof_for(self, triple: Triple) -> Optional[tuple[str, list[Triple]]]:
        return self.derivations.get(_coerce_triple(triple).as_tuple())


# ---------------------------------------------------------------------------
# SymbolicReasoner
# ---------------------------------------------------------------------------
class SymbolicReasoner:
    """Wraps a :class:`DatalogEngine` plus OWL subsumption / consistency logic.

    The reasoner is intentionally tolerant of a missing DatalogEngine: if
    ``neurosym.knowledge.datalog`` cannot be imported (e.g. the parallel
    ``ns-A`` task has not yet shipped all its dependencies) or an instance
    is not supplied to the constructor, the reasoner falls back to its own
    seeded, naive semi-naive-style forward-chaining implementation
    (:class:`_FallbackDatalogEngine`). The derivation log it produces is
    shape-compatible with the contract DatalogEngine's, so proofs are
    constructed identically in either mode.
    """

    OWL_DISJOINT_PREDICATES: frozenset[str] = _DISJOINT_PREDICATES
    RDF_TYPE_PREDICATES: frozenset[str] = _TYPE_PREDICATES
    RDFS_SUBCLASS_PREDICATES: frozenset[str] = _SUBCLASS_PREDICATES

    #: default confidence the reasoner assigns to symbolic derivations
    DEFAULT_CONFIDENCE: float = 0.9

    def __init__(
        self,
        datalog_engine: "DatalogEngine | None" = None,
        rules: Iterable[Rule] | None = None,
        *,
        seed: int = 42,
        max_iterations: int = 100,
    ) -> None:
        self._seed = int(seed)
        self._max_iterations = int(max_iterations)
        self._proof_builder = ProofBuilder()
        self._query_engine = QueryEngine()
        self._rules: list[Rule] = list(rules) if rules else []

        if datalog_engine is not None:
            self._datalog: Any = datalog_engine
            self._fallback_mode = False
        elif _HAS_DATALOG and _ContractDatalogEngine is not None:
            try:
                self._datalog = _ContractDatalogEngine(seed=self._seed)
                self._fallback_mode = False
            except Exception:  # pragma: no cover - defensive
                _LOG.debug("Contract DatalogEngine instantiation failed; using fallback")
                self._datalog = _FallbackDatalogEngine(seed=self._seed)
                self._fallback_mode = True
        else:
            self._datalog = _FallbackDatalogEngine(seed=self._seed)
            self._fallback_mode = True

        # Register any rules the caller supplied.
        if self._rules:
            for r in self._rules:
                self._register_rule_with_engine(r)

    # ------------------------------------------------------------------
    # rule management
    # ------------------------------------------------------------------
    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    @property
    def datalog_engine(self) -> Any:
        return self._datalog

    @property
    def fallback_mode(self) -> bool:
        """True iff the reasoner is using its internal fallback forward-chainer."""
        return self._fallback_mode

    def add_rule(self, rule: Rule) -> "SymbolicReasoner":
        """Register a Datalog rule with both the reasoner and the engine."""
        self._rules.append(rule)
        self._register_rule_with_engine(rule)
        return self

    def add_rules(self, rules: Iterable[Rule]) -> "SymbolicReasoner":
        for r in rules:
            self.add_rule(r)
        return self

    def _register_rule_with_engine(self, rule: Rule) -> None:
        add_rule = getattr(self._datalog, "add_rule", None)
        if callable(add_rule):
            try:
                add_rule(rule)
                return
            except Exception:
                pass
        load_rules = getattr(self._datalog, "load_rules", None)
        if callable(load_rules):
            try:
                load_rules([rule])
                return
            except Exception:
                pass

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def reason(
        self,
        query: "Query | str | dict[str, Any] | None",
        kg: "KnowledgeGraph",
    ) -> ReasoningResult:
        """Forward-chain, then answer ``query`` against the materialised kg.

        Parameters
        ----------
        query:
            A :class:`Query` model (preferred) or a raw string (parsed via
            :class:`QueryEngine`) or a single pattern dict.
        kg:
            The :class:`KnowledgeGraph` to reason over.

        Returns
        -------
        ReasoningResult
            ``query``    - the original query text.
            ``answer``   - ``list[dict[str, str]]`` of variable bindings
            (empty if no matches), or ``"true"``/``"false"`` for ASK-style
            queries.
            ``proof``    - a :class:`ProofTrace` justifying the first
            derived conclusion (or the first matched fact if none derived).
            ``source``   - always ``"symbolic"``.
            ``confidence`` - the symbolic default of 0.9.
        """
        t_start = time.perf_counter()
        query_model = self._normalise_query(query)
        patterns = self._extract_patterns(query_model)

        # Seed the engine with the KG's facts.
        kg_facts = _kg_iter_triples(kg)
        self._seed_facts(kg_facts)

        # Forward-chain to fixpoint.
        forward_chain = getattr(self._datalog, "forward_chain", None)
        derived: list[Triple] = []
        if callable(forward_chain):
            try:
                derived = forward_chain(max_iterations=self._max_iterations) or []
            except TypeError:
                # Engine's forward_chain takes no args.
                try:
                    derived = forward_chain() or []
                except Exception:
                    derived = []

        # Materialise derived facts back into the KG so callers see them.
        for t in derived:
            _kg_add_triple(kg, t.s, t.p, t.o)

        # Answer the query: conjunctive join across all patterns.
        bindings_rows = self._answer_query(patterns)

        # Pick a goal triple for the proof: instantiate the first pattern with
        # the first binding (or, if no bindings, the first pattern itself).
        goal_triple: Triple | None = None
        if patterns:
            first_pat = patterns[0]
            if bindings_rows:
                goal_triple = _instantiate(first_pat, bindings_rows[0])
            else:
                goal_triple = _instantiate(first_pat, {})

        # Build the proof trace from the engine's derivation log.
        derivations = self._get_derivations()
        base_keys = {t.as_tuple() for t in kg_facts}
        proof = (
            self._proof_builder.build_trace(derivations, goal=goal_triple, base_facts=base_keys)
            if goal_triple is not None
            else None
        )

        # Build the answer payload.
        if (
            patterns
            and len(patterns) == 1
            and not is_variable(patterns[0].get("s", ""))
            and not is_variable(patterns[0].get("p", ""))
            and not is_variable(patterns[0].get("o", ""))
        ):
            # Fully-ground pattern — treat as ASK.
            answer: list[dict[str, str]] | str = "true" if bindings_rows else "false"
        else:
            answer = bindings_rows

        latency = (time.perf_counter() - t_start) * 1000.0
        explanation = format_proof(proof) if proof is not None else ""

        notes_parts: list[str] = []
        if self._fallback_mode:
            notes_parts.append("fallback forward-chainer (contract DatalogEngine unavailable)")
        notes_parts.append(f"{len(derived)} derived triple(s)")
        notes_parts.append(f"{len(bindings_rows)} binding row(s)")
        if explanation:
            notes_parts.append("proof: see ReasoningResult.proof / explanation")

        result = ReasoningResult(
            query=query_model.text if hasattr(query_model, "text") else str(query_model),
            answer=answer,
            confidence=self.DEFAULT_CONFIDENCE if bindings_rows else 0.0,
            proof=proof,
            source="symbolic",  # type: ignore[arg-type]
            latency_ms=latency,
            notes="; ".join(notes_parts),
        )
        return result

    # ------------------------------------------------------------------
    # consistency + subsumption
    # ------------------------------------------------------------------
    def check_consistency(self, kg: "KnowledgeGraph") -> bool:
        """Return ``False`` if any disjoint-class contradiction is detected.

        A contradiction is any entity ``X`` for which both ``X rdf:type A``
        and ``X rdf:type B`` hold, where ``A owl:disjointWith B`` (the
        disjointness relation is symmetric — we treat it as such even when
        only the ``A -> B`` direction is asserted in the kg).
        """
        triples = _kg_iter_triples(kg)
        disjoint_pairs: set[frozenset[str]] = set()
        types_by_entity: dict[str, set[str]] = {}
        for t in triples:
            if t.p in _DISJOINT_PREDICATES:
                disjoint_pairs.add(frozenset((t.s, t.o)))
            elif t.p in _TYPE_PREDICATES:
                types_by_entity.setdefault(t.s, set()).add(t.o)

        if not disjoint_pairs or not types_by_entity:
            return True

        for entity, type_set in types_by_entity.items():
            if len(type_set) < 2:
                continue
            for pair in disjoint_pairs:
                members = tuple(pair)
                if len(members) != 2:
                    continue
                a, b = members
                if a in type_set and b in type_set and a != b:
                    _LOG.warning(
                        "Consistency violation: %s is rdf:type %s and rdf:type %s "
                        "while %s owl:disjointWith %s",
                        entity,
                        a,
                        b,
                        a,
                        b,
                    )
                    return False
        return True

    def subsumes(self, class_a: str, class_b: str, kg: "KnowledgeGraph") -> bool:
        """Return ``True`` iff ``class_a`` subsumes ``class_b`` (OWL closure).

        ``class_a`` subsumes ``class_b`` when ``class_b`` is a (reflexive,
        transitive) subclass of ``class_a``.  Concretely:

        * reflexive: ``class_a == class_b`` -> ``True``
        * direct: ``class_b rdfs:subClassOf class_a`` -> ``True``
        * transitive: walking the ``rdfs:subClassOf`` graph upward from
          ``class_b`` reaches ``class_a``.
        """
        if class_a == class_b:
            return True
        triples = _kg_iter_triples(kg)
        parents: dict[str, set[str]] = {}
        for t in triples:
            if t.p in _SUBCLASS_PREDICATES:
                parents.setdefault(t.s, set()).add(t.o)

        seen: set[str] = set()
        queue: deque[str] = deque([class_b])
        while queue:
            cur = queue.popleft()
            if cur == class_a:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for parent in parents.get(cur, ()):
                if parent not in seen:
                    queue.append(parent)
        return False

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _normalise_query(self, query: Any) -> Query:
        if query is None:
            return Query(text="", type=QueryType.DATALOG, params={})
        if isinstance(query, Query):
            return query
        if isinstance(query, str):
            return self._query_engine.build_query(query)
        if isinstance(query, dict):
            # single pattern dict (with s/p/o keys)
            if {"s", "p", "o"} <= set(query.keys()):
                pat = {
                    "s": str(query["s"]),
                    "p": str(query["p"]),
                    "o": str(query["o"]),
                }
                return Query(
                    text=str(query.get("text", "")),
                    type=QueryType.DATALOG,
                    params={"pattern": pat, "patterns": [pat]},
                )
            raise ValueError(f"Unsupported query dict shape: {query!r}")
        raise TypeError(f"Unsupported query type: {type(query).__name__}")

    def _extract_patterns(self, query: Query) -> list[dict[str, str]]:
        params = getattr(query, "params", {}) or {}
        patterns = params.get("patterns")
        if not patterns:
            single = params.get("pattern")
            if single:
                patterns = [single]
            else:
                # fall back to parsing the query text
                if query.text:
                    patterns = self._query_engine.parse_datalog(query.text)
                else:
                    patterns = []
        # Coerce to plain dicts with s/p/o keys
        out: list[dict[str, str]] = []
        for pat in patterns:
            if isinstance(pat, dict):
                out.append(
                    {
                        "s": str(pat.get("s", "?s")),
                        "p": str(pat.get("p", "?p")),
                        "o": str(pat.get("o", "?o")),
                    }
                )
            else:
                # Try to coerce a Triple
                try:
                    t = _coerce_triple(pat)
                    out.append({"s": t.s, "p": t.p, "o": t.o})
                except Exception:
                    continue
        return out

    def _answer_query(self, patterns: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compute the conjunctive join of all query patterns."""
        if not patterns:
            return []
        # Start with all bindings for the first pattern, then join.
        query_fn = getattr(self._datalog, "query", None)
        if not callable(query_fn):
            return []
        try:
            current: list[dict[str, str]] = query_fn(patterns[0])
        except Exception:
            current = []
        if not current:
            return []
        for pat in patterns[1:]:
            next_pat_matches: list[dict[str, str]] = []
            try:
                next_pat_matches = query_fn(pat)
            except Exception:
                next_pat_matches = []
            if not next_pat_matches:
                return []
            joined: list[dict[str, str]] = []
            for b in current:
                for nb in next_pat_matches:
                    merged = self._merge_bindings(b, nb)
                    if merged is not None:
                        joined.append(merged)
            # Deduplicate joined bindings.
            seen: set[tuple] = set()
            deduped: list[dict[str, str]] = []
            for j in joined:
                k = tuple(sorted(j.items()))
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(j)
            current = deduped
            if not current:
                return []
        # Sort for determinism.
        current.sort(key=lambda d: sorted(d.items()))
        return current

    @staticmethod
    def _merge_bindings(a: dict[str, str], b: dict[str, str]) -> dict[str, str] | None:
        """Merge two binding dicts; return None if they conflict."""
        merged = dict(a)
        for k, v in b.items():
            if k in merged:
                if merged[k] != v:
                    return None
            else:
                merged[k] = v
        return merged

    # ------------------------------------------------------------------
    # engine-state helpers
    # ------------------------------------------------------------------
    def _seed_facts(self, facts: Iterable[Triple]) -> None:
        """Seed the engine with a fresh fact table."""
        facts_list = [_coerce_triple(t) for t in facts]
        # Prefer set_facts (bulk replace).
        set_facts = getattr(self._datalog, "set_facts", None)
        if callable(set_facts):
            try:
                set_facts(facts_list)
                return
            except Exception:
                pass
        # Fall back to clear + add_fact loop.
        clear = getattr(self._datalog, "clear_derived", None) or getattr(
            self._datalog, "clear", None
        )
        if callable(clear):
            try:
                clear()
            except Exception:
                pass
        add_fact = getattr(self._datalog, "add_fact", None)
        if not callable(add_fact):
            return
        for t in facts_list:
            try:
                add_fact(t)
            except Exception:
                continue

    def _get_derivations(self) -> dict[tuple[str, str, str], tuple[str, list[Triple]]]:
        """Return the engine's derivation log as a normalised dict."""
        derivations = getattr(self._datalog, "derivations", None)
        if isinstance(derivations, dict):
            out: dict[tuple[str, str, str], tuple[str, list[Triple]]] = {}
            for k, v in derivations.items():
                try:
                    key = _coerce_triple(k).as_tuple() if not isinstance(k, tuple) else k
                    if isinstance(v, tuple) and len(v) >= 2:
                        rule_name = str(v[0])
                        premises = [_coerce_triple(p) for p in v[1]]
                        out[key] = (rule_name, premises)
                    elif isinstance(v, dict):
                        rule_name = str(v.get("rule_name") or v.get("rule") or "rule")
                        premises = [_coerce_triple(p) for p in (v.get("premises") or [])]
                        out[key] = (rule_name, premises)
                except Exception:
                    continue
            return out
        return {}


__all__ = ["SymbolicReasoner"]
