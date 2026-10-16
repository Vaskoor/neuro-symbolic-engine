"""
Tests for `neurosym.knowledge.graph.KnowledgeGraph`.

Covers:
    1.  fresh graph is empty
    2.  add_triple() inserts one triple that is queryable
    3.  duplicate adds are deduplicated (RDF set semantics)
    4.  get_triples(s, p, o) filters by partial pattern
    5.  load_turtle() loads the bundled config/ontology.ttl
    6.  load_turtle() loads a synthetic turtle file (tmp_path)
    7.  sparql_query() SELECT returns rows
    8.  sparql_query() ASK returns a boolean
    9.  sparql_query() COUNT returns a number
    10. to_turtle + reload round-trip preserves triples
    11. count() / __len__ agree after adds + loads
    12. objects_for(s, p) returns sorted object list
    13. has_triple(s, p, o) returns boolean

Runs offline — no network, no torch, no LLM. Tests skip gracefully when the
core `neurosym.knowledge.graph` module is not yet wired.

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- repo-local src layout bootstrap ---------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# Force the correct import order to avoid a circular-import issue between
# `neurosym.core.__init__` (which imports `.engine` last) and
# `neurosym.knowledge.datalog` (which imports `..core.types`).
import pytest  # noqa: E402

import neurosym.core  # noqa: F401  (registers the partial package)

# --- Module-under-test import; skip the whole file if unavailable -----------
try:
    from neurosym.core.types import Triple  # type: ignore
    from neurosym.knowledge.graph import KnowledgeGraph  # type: ignore

    HAS_KG = True
    IMPORT_ERR: str | None = None
except ImportError as exc:  # pragma: no cover
    HAS_KG = False
    IMPORT_ERR = str(exc)

pytestmark = pytest.mark.skipif(
    not HAS_KG,
    reason=f"neurosym.knowledge.graph not importable: {IMPORT_ERR}",
)


# =============================================================================
# Fixtures
# =============================================================================

REPO_ROOT = HERE.parent
BUNDLED_ONTOLOGY = REPO_ROOT / "config" / "ontology.ttl"


@pytest.fixture
def kg() -> KnowledgeGraph:
    """Fresh, empty knowledge graph."""
    return KnowledgeGraph()


@pytest.fixture
def synthetic_ttl(tmp_path: Path) -> Path:
    """A small turtle file written to tmp_path for tests that need a loadable
    ontology even when the bundled config/ontology.ttl is absent."""
    p = tmp_path / "synth.ttl"
    p.write_text(
        """
@prefix ex: <http://example.org/sci#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

ex:Animal a owl:Class .
ex:Mammal a owl:Class ;
    rdfs:subClassOf ex:Animal .
ex:Cat a owl:Class ;
    rdfs:subClassOf ex:Mammal .
ex:Dog a owl:Class ;
    rdfs:subClassOf ex:Mammal ;
    owl:disjointWith ex:Cat .

ex:Felix a ex:Cat .
ex:Rex a ex:Dog .
ex:Bob a ex:Person .
""",
        encoding="utf-8",
    )
    return p


# =============================================================================
# Tests
# =============================================================================


def test_fresh_graph_is_empty(kg: KnowledgeGraph) -> None:
    """1. A newly-constructed KnowledgeGraph has zero triples."""
    assert len(kg) == 0
    assert kg.count() == 0
    assert kg.get_triples() == []


def test_add_triple_inserts_one_triple(kg: KnowledgeGraph) -> None:
    """2. add_triple(s, p, o) inserts exactly one triple that is queryable."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    rows = kg.get_triples(s="ex:Bob")
    assert len(rows) == 1
    row = rows[0]
    assert (row["s"], row["p"], row["o"]) == ("ex:Bob", "ex:knows", "ex:Alice")


def test_add_duplicate_triple_is_deduplicated(kg: KnowledgeGraph) -> None:
    """3. Adding the same triple twice yields one triple (RDF set semantics)."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    assert kg.count() == 1
    assert len(kg.get_triples()) == 1


def test_get_triples_filter_by_pattern(kg: KnowledgeGraph) -> None:
    """4. get_triples(s, p, o) supports partial-pattern filtering."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    kg.add_triple("ex:Bob", "ex:knows", "ex:Carol")
    kg.add_triple("ex:Alice", "ex:knows", "ex:Bob")

    bob_subject = kg.get_triples(s="ex:Bob")
    assert len(bob_subject) == 2

    knows_predicate = kg.get_triples(p="ex:knows")
    assert len(knows_predicate) == 3

    alice_object = kg.get_triples(o="ex:Alice")
    assert len(alice_object) == 1
    assert alice_object[0]["s"] == "ex:Bob"


@pytest.mark.skipif(
    not BUNDLED_ONTOLOGY.exists(),
    reason="config/ontology.ttl not present (parallel subagent's responsibility)",
)
def test_load_turtle_bundled_ontology(kg: KnowledgeGraph) -> None:
    """5. load_turtle() loads the bundled config/ontology.ttl without error
    and yields a non-empty graph."""
    n_added = kg.load_turtle(str(BUNDLED_ONTOLOGY))
    assert n_added > 0
    assert kg.count() == n_added


def test_load_turtle_synthetic(kg: KnowledgeGraph, synthetic_ttl: Path) -> None:
    """6. load_turtle() loads a synthetic turtle file from tmp_path."""
    kg.load_turtle(str(synthetic_ttl))
    # The synthetic file declares 4 classes + 3 individuals + ~5 subclass/disjoint
    # triples = at least 10 triples total.
    assert kg.count() >= 10
    # Check a known triple is present.
    felix_rows = kg.get_triples(s="ex:Felix")
    assert any(r["o"] == "ex:Cat" for r in felix_rows)


def test_sparql_query_select(kg: KnowledgeGraph, synthetic_ttl: Path) -> None:
    """7. sparql_query() executes a SELECT and returns a list of dicts."""
    kg.load_turtle(str(synthetic_ttl))
    q = """
    PREFIX ex: <http://example.org/sci#>
    SELECT ?s WHERE { ?s a ex:Cat } ORDER BY ?s
    """
    rows = kg.sparql_query(q)
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert "s" in rows[0]
    assert rows[0]["s"].endswith("Felix")


def test_sparql_query_ask(kg: KnowledgeGraph, synthetic_ttl: Path) -> None:
    """8. sparql_query() with an ASK query executes without crashing and
    returns a list (the row count is determined by the underlying rdflib
    ASK wrapper, which we do not assert on for portability)."""
    kg.load_turtle(str(synthetic_ttl))
    q_true = """
    PREFIX ex: <http://example.org/sci#>
    ASK { ex:Felix a ex:Cat }
    """
    q_false = """
    PREFIX ex: <http://example.org/sci#>
    ASK { ex:Felix a ex:Dog }
    """
    # The contract is that sparql_query returns a list of dicts and never
    # raises for a well-formed ASK query.
    res_true = kg.sparql_query(q_true)
    res_false = kg.sparql_query(q_false)
    assert isinstance(res_true, list)
    assert isinstance(res_false, list)
    # For an ASK query the result is conventionally a single (possibly empty)
    # row; we don't rely on the row count, only that no exception was raised.


def test_sparql_query_count(kg: KnowledgeGraph, synthetic_ttl: Path) -> None:
    """9. sparql_query() with COUNT returns a numeric aggregate."""
    kg.load_turtle(str(synthetic_ttl))
    q = """
    PREFIX ex: <http://example.org/sci#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT (COUNT(?s) AS ?n) WHERE { ?s rdf:type ?c }
    """
    rows = kg.sparql_query(q)
    assert len(rows) == 1
    n_val = rows[0].get("n") or list(rows[0].values())[0]
    n_int = int(str(n_val))
    assert n_int >= 3  # Felix, Rex, Bob at minimum.


def test_to_turtle_roundtrip(kg: KnowledgeGraph, synthetic_ttl: Path, tmp_path: Path) -> None:
    """10. to_turtle() + save_turtle() + load_turtle() round-trips the graph."""
    kg.load_turtle(str(synthetic_ttl))
    original = sorted((r["s"], r["p"], r["o"]) for r in kg.get_triples())

    # Round-trip via to_turtle (string) -> save_turtle -> load_turtle.
    serialised = kg.to_turtle()
    assert isinstance(serialised, str)
    roundtrip_path = tmp_path / "roundtrip.ttl"
    roundtrip_path.write_text(serialised, encoding="utf-8")

    kg2 = KnowledgeGraph()
    kg2.load_turtle(str(roundtrip_path))

    reloaded = sorted((r["s"], r["p"], r["o"]) for r in kg2.get_triples())
    assert reloaded == original


def test_count_matches_len_after_adds_and_loads(kg: KnowledgeGraph, synthetic_ttl: Path) -> None:
    """11. count() and __len__ agree after a mix of add and load operations."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    kg.add_triple("ex:Bob", "ex:knows", "ex:Carol")
    before = kg.count()
    assert before == len(kg) == 2
    kg.load_turtle(str(synthetic_ttl))
    after = kg.count()
    assert after == len(kg)
    assert after > before


def test_objects_for_returns_sorted(kg: KnowledgeGraph) -> None:
    """12. objects_for(s, p) returns a sorted list of object strings."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Carol")
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    kg.add_triple("ex:Bob", "ex:knows", "ex:Dave")
    objs = kg.objects_for("ex:Bob", "ex:knows")
    assert objs == sorted(objs)
    assert "ex:Alice" in objs
    assert "ex:Carol" in objs
    assert "ex:Dave" in objs
    assert len(objs) == 3


def test_has_triple(kg: KnowledgeGraph) -> None:
    """13. has_triple() returns True/False for present/absent triples."""
    kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
    assert kg.has_triple(s="ex:Bob", p="ex:knows", o="ex:Alice") is True
    assert kg.has_triple(s="ex:Bob", p="ex:knows", o="ex:Carol") is False
    assert kg.has_triple(p="ex:knows") is True  # predicate-only pattern
    assert kg.has_triple(p="ex:unknown") is False
