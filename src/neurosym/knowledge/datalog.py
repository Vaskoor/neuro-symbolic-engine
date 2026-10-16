# SPDX-License-Identifier: MIT
#
# neurosym.knowledge.datalog
# ---------------------------
# A small, deterministic, pure-Python Datalog engine.
#
# The engine is intentionally decoupled from rdflib: it reasons over
# :class:`neurosym.core.types.Triple` objects whose fields are plain strings
# (URIs or lexical literals). The orchestrating engine seeds the fact table
# from the :class:`KnowledgeGraph` (whose ``get_triples`` already compact
# terms to CURIEs/literals), so rule patterns written in YAML (e.g.
# ``ex:authoredBy``) match the same string form the KG produces.
#
# Evaluation strategy
# ~~~~~~~~~~~~~~~~~~~~
# * **Naive semi-fixpoint**: each iteration snapshots the current facts,
#   applies every rule (ordered by descending ``priority``, stable) and
#   adds newly-derived triples to the live fact table.
# * Determinism: facts and bindings are always sorted lexicographically
#   before matching, so two runs over the same input produce identical
#   derivations and identical proof trees.
# * A derivation log (``derivations``) maps each derived triple to the rule
#   name and the premise triples that fired it — the orchestrator uses this
#   to reconstruct :class:`ProofStep` trees.

"""Deterministic Datalog engine with fixpoint forward-chaining and queries."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

import yaml

from ..core.types import Rule, Triple


class DatalogEngine:
    """A rule-based forward-chaining Datalog reasoner."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self.rules: list[Rule] = []
        self._rules_by_name: dict[str, Rule] = {}

        # Live fact table: key -> Triple (both asserted and derived).
        self.facts: dict[tuple[str, str, str], Triple] = {}
        # Derivation log: derived key -> (rule_name, premise Triples).
        self.derivations: dict[tuple[str, str, str], tuple[str, list[Triple]]] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------
    def add_rule(self, rule: Rule) -> None:
        """Add (or replace-by-name) a rule. Rules are kept priority-sorted."""
        if rule.name in self._rules_by_name:
            self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
        self._rules_by_name[rule.name] = rule
        self._sort_rules()

    def _sort_rules(self) -> None:
        # Stable sort by descending priority; ties keep insertion order.
        self.rules.sort(key=lambda r: -r.priority)

    def load_rules(self, path: str | Path) -> int:
        """Load Datalog rules from a YAML file.

        The YAML may be either a list of rule dicts or a mapping with a
        ``rules:`` key. Returns the number of rules loaded.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, list):
            raw_rules = data
        elif isinstance(data, dict):
            raw_rules = data.get("rules", [])
        else:
            raise ValueError(f"Unsupported rules YAML shape: {type(data).__name__}")
        count = 0
        for r in raw_rules:
            self.add_rule(Rule(**r))
            count += 1
        return count

    def list_rules(self) -> list[Rule]:
        return list(self.rules)

    # ------------------------------------------------------------------
    # Fact management
    # ------------------------------------------------------------------
    def set_facts(self, triples: Iterable[Triple]) -> None:
        """Replace the entire fact table with ``triples`` (clears derivations)."""
        self.facts = {}
        self.derivations = {}
        for t in triples:
            self.facts[t.as_tuple()] = t

    def add_fact(self, triple: Triple) -> bool:
        """Add a base fact. Returns ``True`` if it was new."""
        key = triple.as_tuple()
        if key in self.facts:
            return False
        self.facts[key] = triple
        return True

    def add_facts(self, triples: Iterable[Triple]) -> int:
        added = 0
        for t in triples:
            if self.add_fact(t):
                added += 1
        return added

    def get_facts(self) -> list[Triple]:
        """All known facts (asserted + derived), deterministically sorted."""
        return sorted(self.facts.values(), key=lambda t: t.as_tuple())

    def clear_derived(self) -> None:
        """Drop all derived facts and the derivation log (keep base facts)."""
        derived_keys = [k for k in self.facts if k in self.derivations]
        for k in derived_keys:
            del self.facts[k]
        self.derivations = {}

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------
    @staticmethod
    def _is_var(value: Optional[str]) -> bool:
        return value is not None and value.startswith("?")

    def _match_pattern(
        self,
        pattern: dict[str, str],
        fact: Triple,
        bindings: dict[str, str],
    ) -> Optional[dict[str, str]]:
        """Match one triple pattern against one fact under current bindings.

        Returns the (possibly extended) bindings dict, or ``None`` on failure.
        """
        b = dict(bindings)
        for key in ("s", "p", "o"):
            if key not in pattern:
                continue
            val = pattern[key]
            fval = getattr(fact, key)
            if self._is_var(val):
                if val in b:
                    if b[val] != fval:
                        return None
                else:
                    b[val] = fval
            else:
                if val != fval:
                    return None
        return b

    def _match_body(
        self,
        body: list[dict[str, str]],
        facts: list[Triple],
        bindings: dict[str, str],
        premises: list[Triple],
    ) -> list[tuple[dict[str, str], list[Triple]]]:
        """Recursively join all body patterns; return (bindings, premises) tuples."""
        if not body:
            return [(bindings, premises)]
        first, rest = body[0], body[1:]
        results: list[tuple[dict[str, str], list[Triple]]] = []
        for fact in facts:
            b = self._match_pattern(first, fact, bindings)
            if b is not None:
                results.extend(self._match_body(rest, facts, b, premises + [fact]))
        return results

    def _subst(self, value: Optional[str], bindings: dict[str, str]) -> str:
        if value is None:
            return ""
        if self._is_var(value):
            return bindings.get(value, value)
        return value

    def _instantiate(self, head: dict[str, str], bindings: dict[str, str]) -> Triple:
        return Triple(
            s=self._subst(head.get("s"), bindings),
            p=self._subst(head.get("p"), bindings),
            o=self._subst(head.get("o"), bindings),
        )

    # ------------------------------------------------------------------
    # Forward chaining
    # ------------------------------------------------------------------
    def forward_chain(self, max_iterations: int = 100) -> list[Triple]:
        """Apply rules until a fixpoint (or ``max_iterations``) is reached.

        Returns the list of all newly-derived triples (in derivation order).
        The fact table and derivation log are mutated in place.
        """
        newly_derived: list[Triple] = []
        for _iteration in range(max_iterations):
            # Snapshot at iteration start (naive evaluation).
            facts_snapshot = self.get_facts()
            added_this_iteration = 0
            for rule in list(self.rules):
                matches = self._match_body(rule.body, facts_snapshot, {}, [])
                # Deduplicate by derived head key within a single rule firing.
                seen_heads: set[tuple[str, str, str]] = set()
                for bindings, premises in matches:
                    head_triple = self._instantiate(rule.head, bindings)
                    key = head_triple.as_tuple()
                    if key in seen_heads:
                        continue
                    seen_heads.add(key)
                    if key not in self.facts:
                        self.facts[key] = head_triple
                        # Record derivation (use a sorted, deduplicated premise list
                        # so the proof tree is stable).
                        self.derivations[key] = (
                            rule.name,
                            self._dedupe_premises(premises),
                        )
                        newly_derived.append(head_triple)
                        added_this_iteration += 1
            if added_this_iteration == 0:
                break
        return newly_derived

    @staticmethod
    def _dedupe_premises(premises: list[Triple]) -> list[Triple]:
        """Deduplicate premises while preserving first-seen order."""
        seen: set[tuple[str, str, str]] = set()
        out: list[Triple] = []
        for p in premises:
            k = p.as_tuple()
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query(self, pattern: dict[str, str]) -> list[dict[str, str]]:
        """Return all variable bindings satisfying ``pattern`` against all facts."""
        facts = self.get_facts()
        out: list[dict[str, str]] = []
        for fact in facts:
            b = self._match_pattern(pattern, fact, {})
            if b is not None:
                out.append(dict(b))
        # Deterministic ordering by binding items.
        out.sort(key=lambda d: sorted(d.items()))
        return out

    def query_conclusion(self, pattern: dict[str, str], binding: dict[str, str]) -> Triple:
        """Instantiate ``pattern`` with a binding to recover the matched triple."""
        return Triple(
            s=self._subst(pattern.get("s"), binding),
            p=self._subst(pattern.get("p"), binding),
            o=self._subst(pattern.get("o"), binding),
        )

    # ------------------------------------------------------------------
    # Proof reconstruction
    # ------------------------------------------------------------------
    def proof_for(self, triple: Triple) -> Optional[tuple[str, list[Triple]]]:
        """Return ``(rule_name, premises)`` for a derived triple, or ``None``."""
        return self.derivations.get(triple.as_tuple())


__all__ = ["DatalogEngine"]
