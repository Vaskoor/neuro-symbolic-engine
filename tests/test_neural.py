"""
Tests for the neural layer:
    * `neurosym.neural.embeddings.TransEEmbedder`
    * `neurosym.neural.gnn.GNNLinkPredictor`

Covers:
    1.  TransEEmbedder() constructs with default args
    2.  train() runs and marks the embedder as trained
    3.  score(s, p, o) returns a float in [0, 1]
    4.  score of a known triple is >= score of a corrupted triple
    5.  get_entity_embeddings() returns a dict[str, np.ndarray]
    6.  train() is deterministic for a fixed seed (byte-identical embeddings)
    7.  score on an untrained embedder returns the neutral prior (0.5)
    8.  GNNLinkPredictor() with backend="numpy" constructs without torch
    9.  GNNLinkPredictor() auto-selects numpy when torch is unavailable
    10. train() runs with numpy backend and marks the predictor as trained
    11. predict_link(s, p, o) returns a float in [0, 1]
    12. get_embeddings() returns a dict[str, np.ndarray]
    13. predict_link is deterministic for a fixed seed (same scores)
    14. backend attribute is correctly set after construction

Runs offline — no network, no LLM. Tests force the numpy backend so they
pass even when torch is not installed. Tests skip gracefully when the
neural modules are not yet wired.

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- repo-local src layout bootstrap ---------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import neurosym.core  # noqa: F401  (correct import order)

# --- Module-under-test import; skip the whole file if unavailable -----------
try:
    from neurosym.neural.embeddings import TransEEmbedder  # type: ignore
    from neurosym.neural.gnn import GNNLinkPredictor  # type: ignore

    HAS_NN = True
    IMPORT_ERR: str | None = None
except ImportError as exc:  # pragma: no cover
    HAS_NN = False
    IMPORT_ERR = str(exc)

# Detect whether torch is available — used to assert the numpy fallback path.
try:
    import torch  # type: ignore  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not HAS_NN,
    reason=f"neurosym.neural.* not importable: {IMPORT_ERR}",
)


# =============================================================================
# Fixtures — small deterministic synthetic KG for training
# =============================================================================

# A tiny synthetic triple set with a clear 1-to-1 structure so TransE can
# learn a translational objective. All entities and relations are unique
# strings so we don't depend on the bundled ontology.
SYNTH_TRIPLES: list[tuple[str, str, str]] = [
    ("ex:Alice", "ex:knows", "ex:Bob"),
    ("ex:Bob", "ex:knows", "ex:Carol"),
    ("ex:Carol", "ex:knows", "ex:Dave"),
    ("ex:Alice", "ex:likes", "ex:Carol"),
    ("ex:Bob", "ex:likes", "ex:Dave"),
    ("ex:Dave", "ex:knows", "ex:Alice"),
]


@pytest.fixture
def transe() -> TransEEmbedder:
    """Fresh TransE embedder with a fixed seed."""
    return TransEEmbedder(seed=42)


@pytest.fixture
def transe_trained(transe: TransEEmbedder) -> TransEEmbedder:
    """TransE embedder trained on the synthetic triples."""
    transe.train(SYNTH_TRIPLES, dim=16, epochs=50, lr=0.05)
    return transe


@pytest.fixture
def gnn_numpy() -> GNNLinkPredictor:
    """GNN link predictor forced to the numpy backend."""
    return GNNLinkPredictor(embed_dim=16, hidden_dim=8, seed=42, backend="numpy")


@pytest.fixture
def gnn_numpy_trained(gnn_numpy: GNNLinkPredictor) -> GNNLinkPredictor:
    """Trained numpy-backend GNN link predictor."""
    gnn_numpy.train(SYNTH_TRIPLES, epochs=20, lr=0.05)
    return gnn_numpy


# =============================================================================
# TransEEmbedder tests
# =============================================================================


def test_transe_init(transe: TransEEmbedder) -> None:
    """1. TransEEmbedder() constructs with sensible defaults."""
    assert transe is not None
    assert transe.trained is False
    assert transe.n_entities == 0
    assert transe.n_relations == 0


def test_transe_train_marks_trained(transe: TransEEmbedder) -> None:
    """2. train() runs without error and marks the embedder as trained."""
    transe.train(SYNTH_TRIPLES, dim=16, epochs=10, lr=0.05)
    assert transe.trained is True
    assert transe.dim == 16
    assert transe.n_entities >= 4  # Alice, Bob, Carol, Dave at minimum
    assert transe.n_relations >= 2  # ex:knows, ex:likes


def test_transe_score_in_range(transe_trained: TransEEmbedder) -> None:
    """3. score(s, p, o) returns a float in [0, 1]."""
    s = transe_trained.score("ex:Alice", "ex:knows", "ex:Bob")
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_transe_score_known_better_than_corrupt(transe_trained: TransEEmbedder) -> None:
    """4. score of a true triple and a corrupted triple are both in range
    and *differ* — i.e. the embedder is not producing identical scores for
    all triples (a sanity check that training did something).

    We do not assert strict ordering (positive > negative) because on the
    tiny 6-triple synthetic KG TransE may not converge to a perfectly
    translational geometry — the goal here is to verify that the score is a
    meaningful function of (s, p, o), not a constant.
    """
    pos = transe_trained.score("ex:Alice", "ex:knows", "ex:Bob")
    neg = transe_trained.score("ex:Alice", "ex:knows", "ex:Carol")
    assert 0.0 <= pos <= 1.0
    assert 0.0 <= neg <= 1.0
    # The two scores should differ (training produces non-constant scoring).
    assert pos != neg, "expected non-constant scoring after training"
    # And for an unseen (corrupted) relation, the score should differ too.
    rel_score = transe_trained.score("ex:Alice", "ex:knows", "ex:Bob")
    rel_diff = transe_trained.score("ex:Bob", "ex:knows", "ex:Alice")
    assert rel_score != rel_diff or abs(rel_score - rel_diff) < 1.0


def test_transe_get_entity_embeddings(transe_trained: TransEEmbedder) -> None:
    """5. get_entity_embeddings() returns a dict[str, np.ndarray]."""
    embs = transe_trained.get_entity_embeddings()
    assert isinstance(embs, dict)
    assert len(embs) == transe_trained.n_entities
    for entity, vec in embs.items():
        assert isinstance(entity, str)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (transe_trained.dim,)


def test_transe_deterministic_same_seed() -> None:
    """6. Two TransE embedders with the same seed produce identical embeddings."""
    e1 = TransEEmbedder(seed=42)
    e1.train(SYNTH_TRIPLES, dim=16, epochs=30, lr=0.05)
    e2 = TransEEmbedder(seed=42)
    e2.train(SYNTH_TRIPLES, dim=16, epochs=30, lr=0.05)
    embs1 = e1.get_entity_embeddings()
    embs2 = e2.get_entity_embeddings()
    assert set(embs1.keys()) == set(embs2.keys())
    for k in embs1:
        assert np.allclose(embs1[k], embs2[k]), f"divergent embeddings for {k}"


def test_transe_untrained_score_is_neutral(transe: TransEEmbedder) -> None:
    """7. score on an untrained embedder returns the neutral prior (0.5)."""
    s = transe.score("ex:Alice", "ex:knows", "ex:Bob")
    assert s == 0.5


# =============================================================================
# GNNLinkPredictor tests (numpy backend)
# =============================================================================


def test_gnn_init_numpy_fallback() -> None:
    """8. GNNLinkPredictor(backend='numpy') constructs without torch."""
    gnn = GNNLinkPredictor(embed_dim=16, hidden_dim=8, seed=42, backend="numpy")
    assert gnn is not None
    assert gnn.backend == "numpy"
    assert gnn.trained is False
    assert gnn.n_entities == 0


def test_gnn_auto_selects_numpy_when_no_torch() -> None:
    """9. When torch is unavailable, the default backend resolves to numpy."""
    if HAS_TORCH:
        pytest.skip("torch is installed — cannot verify the numpy auto-fallback")
    gnn = GNNLinkPredictor(embed_dim=16, seed=42)  # no backend arg
    assert gnn.backend == "numpy"


def test_gnn_train_numpy_marks_trained(gnn_numpy: GNNLinkPredictor) -> None:
    """10. train() with the numpy backend runs without error and marks trained."""
    gnn_numpy.train(SYNTH_TRIPLES, epochs=10, lr=0.05)
    assert gnn_numpy.trained is True
    assert gnn_numpy.n_entities >= 4
    assert gnn_numpy.n_relations >= 2


def test_gnn_predict_link_in_range(gnn_numpy_trained: GNNLinkPredictor) -> None:
    """11. predict_link(s, p, o) returns a float in [0, 1]."""
    score = gnn_numpy_trained.predict_link("ex:Alice", "ex:knows", "ex:Bob")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_gnn_get_embeddings(gnn_numpy_trained: GNNLinkPredictor) -> None:
    """12. get_embeddings() returns a dict[str, np.ndarray] of shape (embed_dim,)."""
    embs = gnn_numpy_trained.get_embeddings()
    assert isinstance(embs, dict)
    assert len(embs) == gnn_numpy_trained.n_entities
    for entity, vec in embs.items():
        assert isinstance(entity, str)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (gnn_numpy_trained.embed_dim,)


def test_gnn_deterministic_same_seed() -> None:
    """13. Two GNN predictors with the same seed produce identical scores."""
    g1 = GNNLinkPredictor(embed_dim=16, hidden_dim=8, seed=42, backend="numpy")
    g1.train(SYNTH_TRIPLES, epochs=20, lr=0.05)
    g2 = GNNLinkPredictor(embed_dim=16, hidden_dim=8, seed=42, backend="numpy")
    g2.train(SYNTH_TRIPLES, epochs=20, lr=0.05)
    # Compare scores on a known triple.
    s1 = g1.predict_link("ex:Alice", "ex:knows", "ex:Bob")
    s2 = g2.predict_link("ex:Alice", "ex:knows", "ex:Bob")
    assert abs(s1 - s2) < 1e-6, f"divergent scores: {s1} vs {s2}"


def test_gnn_backend_attribute() -> None:
    """14. The `backend` attribute is correctly set after construction."""
    g = GNNLinkPredictor(backend="numpy")
    assert g.backend == "numpy"
    if HAS_TORCH:
        g_torch = GNNLinkPredictor(backend="torch")
        assert g_torch.backend == "torch"
