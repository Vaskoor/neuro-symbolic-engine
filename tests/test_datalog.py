"""
Tests for `neurosym.knowledge.datalog.DatalogEngine`.

Covers:
    1.  fresh engine starts with zero rules and zero facts
    2.  add_rule() registers a rule that fires
    3.  forward_chain() derives expected facts
    4.  query() with variables returns matching bindings
    5.  fixpoint convergence — running forward_chain twice is idempotent
    6.  load_rules() loads the bundled config/rules.yaml (10 rules)
    7.  Rule 1 — derive_has_institution fires
    8.  Rule 2 — derive_has_institution_employed fires
    9.  Rule 3 — derive_indirectly_uses_method fires
    10. Rule 4 — derive_researcher_field fires
    11. Rule 5 — derive_method_expert fires
    12. Rule 6 — derive_high_impact fires
    13. Rule 7 — derive_high_impact_researcher fires (chained)
    14. Rule 8 — propagate_indirect_method transitive propagation
    15. Rule 9 — derive_collaborated (chained from rule 5)
    16. Rule 10 — transitive_cites (transitive closure)
    17. proof_for() returns the derivation chain for a derived fact

Runs offline — no network, no torch, no LLM. Tests skip gracefully when the
core `neurosym.knowledge.datalog` module is not yet wired.

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""

from __future__ import annotations  # noqa: F401

import sys  # noqa: F401
from pathlib import Path  # noqa: F401

# --- repo-local src layout bootstrap ---------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# Force the correct import order to avoid the circular-import issue between
# `neurosym.core.__init__` (which imports `.engine` last) and
# `neurosym.knowledge.datalog` (which imports `..core.types`).
import pytest  # noqa: E402

import neurosym.core  # noqa: F401

# --- Module-under-test import; skip the whole file if unavailable -----------
try:
    from neurosym.core.types import Rule, Triple  # type: ignore  # noqa: F401
    from neurosym.knowledge.datalog import DatalogEngine  # type: ignore  # noqa: F401

    HAS_DL = True
    IMPORT_ERR: str | None = None
except ImportError as exc:  # pragma: no cover
    HAS_DL = False
    IMPORT_ERR = str(exc)

pytestmark = pytest.mark.skipif(
    not HAS_DL,
    reason=f"neurosym.knowledge.datalog not importable: {IMPORT_ERR}",
)


# =============================================================================
# Fixtures
# =============================================================================

REPO_ROOT = HERE.parent
BUNDLED_RULES = REPO_ROOT / "config" / "rules.yaml"


@pytest.fixture
def dl() -> DatalogEngine:
    """Fresh, empty Datalog engine with deterministic seed."""
    return DatalogEngine(seed=42)


def _t(s: str, p: str, o: str) -> Triple:
    """Shorthand to build a Triple."""
    return Triple(s=s, p=p, o=o)


def _seed_facts(dl: DatalogEngine, facts: list[tuple[str, str, str]]) -> None:
    """Bulk-add base facts (as Triples) to the engine."""
    dl.set_facts([_t(s, p, o) for (s, p, o) in facts])


# =============================================================================
# Basic tests
# =============================================================================


def test_fresh_engine_is_empty(dl: DatalogEngine) -> None:
    """1. A newly-constructed DatalogEngine has zero rules and zero facts."""
    assert len(dl.rules) == 0
    assert len(dl.facts) == 0
    assert dl.get_facts() == []


def test_add_rule_fires(dl: DatalogEngine) -> None:
    """2. add_rule() registers a rule that fires on a matching fact set."""
    dl.add_rule(
        Rule(
            name="ancestor",
            body=[{"s": "?x", "p": "ex:parent", "o": "?y"}],
            head={"s": "?x", "p": "ex:ancestor", "o": "?y"},
        )
    )
    _seed_facts(dl, [("ex:Bob", "ex:parent", "ex:Alice")])
    derived = dl.forward_chain()
    assert len(derived) == 1
    assert derived[0].as_tuple() == ("ex:Bob", "ex:ancestor", "ex:Alice")


def test_forward_chain_basic(dl: DatalogEngine) -> None:
    """3. forward_chain() returns the list of newly-derived triples."""
    dl.add_rule(
        Rule(
            name="uncle",
            body=[
                {"s": "?x", "p": "ex:parent", "o": "?y"},
                {"s": "?y", "p": "ex:brother", "o": "?z"},
            ],
            head={"s": "?x", "p": "ex:uncle", "o": "?z"},
        )
    )
    _seed_facts(
        dl,
        [
            ("ex:Bob", "ex:parent", "ex:Alice"),
            ("ex:Alice", "ex:brother", "ex:Charlie"),
        ],
    )
    derived = dl.forward_chain()
    assert any(t.as_tuple() == ("ex:Bob", "ex:uncle", "ex:Charlie") for t in derived)


def test_query_with_variables(dl: DatalogEngine) -> None:
    """4. query(pattern) returns all variable bindings."""
    _seed_facts(
        dl,
        [
            ("ex:Bob", "ex:knows", "ex:Alice"),
            ("ex:Bob", "ex:knows", "ex:Carol"),
            ("ex:Alice", "ex:knows", "ex:Bob"),
        ],
    )
    pattern = {"s": "ex:Bob", "p": "ex:knows", "o": "?y"}
    bindings = dl.query(pattern)
    assert len(bindings) == 2
    objects = sorted(b["?y"] for b in bindings)
    assert objects == ["ex:Alice", "ex:Carol"]


def test_fixpoint_convergence_idempotent(dl: DatalogEngine) -> None:
    """5. forward_chain() is idempotent — calling it twice yields the same
    fact set on the second call (no new derivations)."""
    dl.add_rule(
        Rule(
            name="ancestor_tc",
            body=[
                {"s": "?x", "p": "ex:parent", "o": "?y"},
                {"s": "?y", "p": "ex:ancestor", "o": "?z"},
            ],
            head={"s": "?x", "p": "ex:ancestor", "o": "?z"},
        )
    )
    dl.add_rule(
        Rule(
            name="ancestor_base",
            body=[{"s": "?x", "p": "ex:parent", "o": "?y"}],
            head={"s": "?x", "p": "ex:ancestor", "o": "?y"},
        )
    )
    _seed_facts(
        dl,
        [
            ("ex:Bob", "ex:parent", "ex:Alice"),
            ("ex:Alice", "ex:parent", "ex:Carol"),
        ],
    )
    first = dl.forward_chain()
    n_after_first = len(dl.facts)
    second = dl.forward_chain()
    n_after_second = len(dl.facts)
    assert len(first) > 0, "first forward_chain should derive new facts"
    assert len(second) == 0, "second forward_chain should derive nothing new"
    assert n_after_first == n_after_second


@pytest.mark.skipif(
    not BUNDLED_RULES.exists(),
    reason="config/rules.yaml not present (parallel subagent's responsibility)",
)
def test_load_rules_bundled_yaml(dl: DatalogEngine) -> None:
    """6. load_rules() loads all 10 rules from config/rules.yaml."""
    n = dl.load_rules(str(BUNDLED_RULES))
    assert n == 10
    assert len(dl.rules) == 10
    rule_names = {r.name for r in dl.rules}
    expected = {
        "derive_has_institution",
        "derive_has_institution_employed",
        "derive_indirectly_uses_method",
        "derive_researcher_field",
        "derive_method_expert",
        "derive_high_impact",
        "derive_high_impact_researcher",
        "propagate_indirect_method",
        "derive_collaborated",
        "transitive_cites",
    }
    assert expected.issubset(rule_names)


# =============================================================================
# Per-rule tests — load the 10 bundled rules and verify each derives its
# expected fact from synthetic instance data.
# =============================================================================


@pytest.fixture
def dl_with_rules(dl: DatalogEngine) -> DatalogEngine:
    """A DatalogEngine with the bundled rules loaded."""
    if BUNDLED_RULES.exists():
        dl.load_rules(str(BUNDLED_RULES))
    else:  # pragma: no cover
        pytest.skip("config/rules.yaml not present")
    return dl


def _has_fact(dl: DatalogEngine, s: str, p: str, o: str) -> bool:
    return _t(s, p, o).as_tuple() in dl.facts


def test_rule1_derive_has_institution(dl_with_rules: DatalogEngine) -> None:
    """7. Rule 1: paper X authoredBy R, R affiliatedWith I → X hasInstitution I."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P1", "ex:authoredBy", "ex:R1"),
            ("ex:R1", "ex:affiliatedWith", "ex:I1"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:P1", "ex:hasInstitution", "ex:I1")


def test_rule2_derive_has_institution_employed(dl_with_rules: DatalogEngine) -> None:
    """8. Rule 2: paper X authoredBy R, R employedBy I → X hasInstitution I."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P2", "ex:authoredBy", "ex:R2"),
            ("ex:R2", "ex:employedBy", "ex:I2"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:P2", "ex:hasInstitution", "ex:I2")


def test_rule3_derive_indirectly_uses_method(dl_with_rules: DatalogEngine) -> None:
    """9. Rule 3: P3 cites P4, P4 usesMethod M → P3 indirectlyUsesMethod M."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P3", "ex:cites", "ex:P4"),
            ("ex:P4", "ex:usesMethod", "ex:M1"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:P3", "ex:indirectlyUsesMethod", "ex:M1")


def test_rule4_derive_researcher_field(dl_with_rules: DatalogEngine) -> None:
    """10. Rule 4: P5 authoredBy R, P5 belongsToField F → R worksInField F."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P5", "ex:authoredBy", "ex:R3"),
            ("ex:P5", "ex:belongsToField", "ex:F1"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:R3", "ex:worksInField", "ex:F1")


def test_rule5_derive_method_expert(dl_with_rules: DatalogEngine) -> None:
    """11. Rule 5: P6 authoredBy R, P6 usesMethod M → R knowsMethod M."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P6", "ex:authoredBy", "ex:R4"),
            ("ex:P6", "ex:usesMethod", "ex:M2"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:R4", "ex:knowsMethod", "ex:M2")


def test_rule6_derive_high_impact(dl_with_rules: DatalogEngine) -> None:
    """12. Rule 6: P7 publishedIn V, V isTopVenue 'true' → P7 isHighImpact 'true'."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P7", "ex:publishedIn", "ex:V1"),
            ("ex:V1", "ex:isTopVenue", "true"),
        ],
    )
    dl_with_rules.forward_chain()
    assert _has_fact(dl_with_rules, "ex:P7", "ex:isHighImpact", "true")


def test_rule7_derive_high_impact_researcher(dl_with_rules: DatalogEngine) -> None:
    """13. Rule 7: P8 authoredBy R5, P8 isHighImpact 'true' → R5 hasHighImpactPaper 'true'.

    Depends on rule 6 firing for P8 first — exercises chained inference.
    """
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P8", "ex:authoredBy", "ex:R5"),
            ("ex:P8", "ex:publishedIn", "ex:V2"),
            ("ex:V2", "ex:isTopVenue", "true"),
        ],
    )
    dl_with_rules.forward_chain()
    # First rule 6 derives isHighImpact, then rule 7 derives hasHighImpactPaper.
    assert _has_fact(dl_with_rules, "ex:P8", "ex:isHighImpact", "true")
    assert _has_fact(dl_with_rules, "ex:R5", "ex:hasHighImpactPaper", "true")


def test_rule8_propagate_indirect_method(dl_with_rules: DatalogEngine) -> None:
    """14. Rule 8: P9 cites P3, P3 indirectlyUsesMethod M1 (derived by rule 3)
    → P9 indirectlyUsesMethod M1.

    Exercises transitive propagation across fixpoint iterations.
    """
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P3", "ex:cites", "ex:P4"),
            ("ex:P4", "ex:usesMethod", "ex:M1"),
            ("ex:P9", "ex:cites", "ex:P3"),
        ],
    )
    dl_with_rules.forward_chain()
    # Rule 3 derives P3 indirectlyUsesMethod M1, then rule 8 propagates to P9.
    assert _has_fact(dl_with_rules, "ex:P3", "ex:indirectlyUsesMethod", "ex:M1")
    assert _has_fact(dl_with_rules, "ex:P9", "ex:indirectlyUsesMethod", "ex:M1")


def test_rule9_derive_collaborated(dl_with_rules: DatalogEngine) -> None:
    """15. Rule 9: R6 knowsMethod M3, R7 knowsMethod M3, P10 authoredBy R6,
    P11 authoredBy R7, P10 cites P11 → R6 collaboratedWith R7.

    Depends on rule 5 firing for R6 and R7.
    """
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P10", "ex:authoredBy", "ex:R6"),
            ("ex:P10", "ex:usesMethod", "ex:M3"),
            ("ex:P11", "ex:authoredBy", "ex:R7"),
            ("ex:P11", "ex:usesMethod", "ex:M3"),
            ("ex:P10", "ex:cites", "ex:P11"),
        ],
    )
    dl_with_rules.forward_chain()
    # Rule 5 derives knowsMethod, then rule 9 derives collaboratedWith.
    assert _has_fact(dl_with_rules, "ex:R6", "ex:knowsMethod", "ex:M3")
    assert _has_fact(dl_with_rules, "ex:R7", "ex:knowsMethod", "ex:M3")
    assert _has_fact(dl_with_rules, "ex:R6", "ex:collaboratedWith", "ex:R7")


def test_rule10_transitive_cites(dl_with_rules: DatalogEngine) -> None:
    """16. Rule 10: transitive closure of citation.

    X cites Y, Y cites Z → X cites Z. Verifies the fixpoint produces the
    transitive closure across multiple iterations.
    """
    _seed_facts(
        dl_with_rules,
        [
            ("ex:A", "ex:cites", "ex:B"),
            ("ex:B", "ex:cites", "ex:C"),
            ("ex:C", "ex:cites", "ex:D"),
        ],
    )
    dl_with_rules.forward_chain()
    # After fixpoint, A should cite B, C, D; B should cite C, D; C should cite D.
    assert _has_fact(dl_with_rules, "ex:A", "ex:cites", "ex:C")
    assert _has_fact(dl_with_rules, "ex:A", "ex:cites", "ex:D")
    assert _has_fact(dl_with_rules, "ex:B", "ex:cites", "ex:D")


def test_proof_for_derived_fact(dl_with_rules: DatalogEngine) -> None:
    """17. proof_for(triple) returns the (rule_name, premises) for a derived fact."""
    _seed_facts(
        dl_with_rules,
        [
            ("ex:P1", "ex:authoredBy", "ex:R1"),
            ("ex:R1", "ex:affiliatedWith", "ex:I1"),
        ],
    )
    dl_with_rules.forward_chain()
    derived = _t("ex:P1", "ex:hasInstitution", "ex:I1")
    proof = dl_with_rules.proof_for(derived)
    assert proof is not None, "derived fact should have a proof"
    rule_name, premises = proof
    assert rule_name == "derive_has_institution"
    assert len(premises) == 2
    premise_keys = [p.as_tuple() for p in premises]
    assert ("ex:P1", "ex:authoredBy", "ex:R1") in premise_keys
    assert ("ex:R1", "ex:affiliatedWith", "ex:I1") in premise_keys
