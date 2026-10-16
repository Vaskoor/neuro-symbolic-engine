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
"""TransE knowledge-graph embeddings for the neuro-symbolic engine.

:class:`TransEEmbedder` learns entity and relation embeddings under the
TransE objective: for every true triple ``(h, r, t)`` we want
``h + r ≈ t`` (in L2 distance). Embeddings are learned in pure numpy with
vanilla SGD on the margin-ranking loss

    loss = max(0, margin - d(h_pos + r, t_pos) + d(h_neg + r, t_neg))

where negatives are produced by corrupting either the head or the tail of a
positive triple.

The embedder is **fully deterministic**: every random draw flows through a
seeded ``numpy.random.RandomState``, so two ``TransEEmbedder`` instances
trained on the same triples with the same seed produce byte-identical
embeddings and scores.

The :meth:`score` method returns a confidence in ``[0, 1]`` derived from the
margin-based distance via a sigmoid, so it can be plugged into the same
"link-trust" pipeline as :class:`neurosym.neural.gnn.GNNLinkPredictor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

try:  # contract from sibling task ns-A
    from neurosym.core.types import Triple
except ImportError:  # pragma: no cover - fallback
    Triple = None  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Triple coercion helper (mirrors the one in gnn.py — kept local to keep
# this module importable in isolation).
# ---------------------------------------------------------------------------
def _triple_as_spo(triple: Any) -> tuple[str, str, str]:
    """Return the ``(s, p, o)`` string triple from any triple-like value."""
    if triple is None:
        raise ValueError("Cannot extract (s, p, o) from None")
    if isinstance(triple, Mapping):
        if {"s", "p", "o"} <= set(triple.keys()):
            return str(triple["s"]), str(triple["p"]), str(triple["o"])
        if {"subject", "predicate", "object"} <= set(triple.keys()):
            return str(triple["subject"]), str(triple["predicate"]), str(triple["object"])
    if isinstance(triple, (tuple, list)) and len(triple) >= 3:
        return str(triple[0]), str(triple[1]), str(triple[2])
    if hasattr(triple, "s") and hasattr(triple, "p") and hasattr(triple, "o"):
        return str(getattr(triple, "s")), str(getattr(triple, "p")), str(getattr(triple, "o"))
    if hasattr(triple, "subject") and hasattr(triple, "predicate") and hasattr(triple, "object"):
        return (
            str(getattr(triple, "subject")),
            str(getattr(triple, "predicate")),
            str(getattr(triple, "object")),
        )
    raise TypeError(f"Cannot extract (s, p, o) from {type(triple).__name__!r}")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise each row of ``x`` (entity embeddings are unit-norm in TransE)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return x / norm


class TransEEmbedder:
    """TransE knowledge-graph embedding (h + r ≈ t).

    Parameters
    ----------
    seed:
        Seed for the deterministic ``numpy.random.RandomState`` used both
        for initialisation and for negative sampling.
    margin:
        Margin hyper-parameter of the ranking loss. Defaults to ``1.0``,
        the canonical TransE value.
    gamma:
        Softmax temperature for :meth:`score` — the higher, the sharper the
        sigmoid confidence output.
    """

    def __init__(
        self,
        seed: int = 42,
        margin: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.margin = float(margin)
        self.gamma = float(gamma)
        self._rng = np.random.RandomState(self.seed)

        # Vocabulary
        self._entity_to_idx: dict[str, int] = {}
        self._idx_to_entity: list[str] = []
        self._relation_to_idx: dict[str, int] = {}
        self._idx_to_relation: list[str] = []

        # Embeddings (filled by train)
        self._entity_emb: np.ndarray | None = None
        self._relation_emb: np.ndarray | None = None
        self._dim: int = 0
        self._trained: bool = False
        self._epochs_run: int = 0
        self._lr_used: float = 0.0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_entities(self) -> int:
        return len(self._idx_to_entity)

    @property
    def n_relations(self) -> int:
        return len(self._idx_to_relation)

    # ------------------------------------------------------------------
    # vocabulary construction
    # ------------------------------------------------------------------
    def _build_vocab(self, triples: Sequence[tuple[str, str, str]]) -> None:
        ents: list[str] = []
        rels: list[str] = []
        seen_e: set[str] = set()
        seen_r: set[str] = set()
        for s, p, o in triples:
            for e in (s, o):
                if e not in seen_e:
                    seen_e.add(e)
                    ents.append(e)
            if p not in seen_r:
                seen_r.add(p)
                rels.append(p)
        # Deterministic sorting — same triples -> same vocab order.
        ents.sort()
        rels.sort()
        self._idx_to_entity = ents
        self._entity_to_idx = {e: i for i, e in enumerate(ents)}
        self._idx_to_relation = rels
        self._relation_to_idx = {r: i for i, r in enumerate(rels)}

    def _init_embeddings(self, dim: int) -> None:
        n = max(self.n_entities, 1)
        m = max(self.n_relations, 1)
        # Uniform [-1, 1] / sqrt(dim) initialisation (a common TransE choice).
        bound = 1.0 / np.sqrt(max(dim, 1))
        self._entity_emb = self._rng.uniform(-bound, bound, size=(n, dim)).astype(np.float64)
        self._relation_emb = self._rng.uniform(-bound, bound, size=(m, dim)).astype(np.float64)
        # L2-normalise embeddings (canonical TransE constraint).
        self._entity_emb = _l2_normalize(self._entity_emb)
        self._relation_emb = _l2_normalize(self._relation_emb)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(
        self,
        triples: Iterable[Any],
        dim: int = 50,
        epochs: int = 100,
        lr: float = 0.01,
        neg_samples: int = 1,
        verbose: bool = False,
    ) -> "TransEEmbedder":
        """Train the TransE embeddings in pure numpy.

        Parameters
        ----------
        triples:
            Iterable of triple-like values (Triple models, ``(s, p, o)``
            tuples, or dicts with ``s``/``p``/``o`` keys).
        dim:
            Embedding dimensionality.
        epochs:
            Number of full passes over the training triples.
        lr:
            Learning rate for vanilla SGD.
        neg_samples:
            Number of negative samples (corrupted head or tail) drawn per
            positive triple per epoch.
        verbose:
            When True, prints the running loss to stderr at the end of each
            10th epoch.

        Returns
        -------
        TransEEmbedder
            ``self`` (for chaining).
        """
        spo_list: list[tuple[str, str, str]] = []
        for t in triples:
            try:
                spo_list.append(_triple_as_spo(t))
            except Exception:
                continue
        if not spo_list:
            self._trained = True
            self._dim = int(dim)
            return self

        # Deduplicate (deterministic).
        seen: set[tuple[str, str, str]] = set()
        deduped: list[tuple[str, str, str]] = []
        for s, p, o in spo_list:
            key = (s, p, o)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        deduped.sort()

        self._dim = int(dim)
        self._build_vocab(deduped)
        self._init_embeddings(self._dim)

        pos_idx = [
            (self._entity_to_idx[s], self._relation_to_idx[p], self._entity_to_idx[o])
            for s, p, o in deduped
        ]
        n_ent = self.n_entities

        for epoch in range(int(epochs)):
            total_loss = 0.0
            n_examples = 0
            for si, pi, oi in pos_idx:
                h = self._entity_emb[si]
                r = self._relation_emb[pi]
                t = self._entity_emb[oi]
                # Positive distance
                d_pos = float(np.linalg.norm(h + r - t))
                # Generate one or more negatives by corrupting head OR tail.
                for _ in range(neg_samples):
                    # Alternate between head and tail corruption
                    if self._rng.rand() < 0.5:
                        # corrupt head
                        nh = int(self._rng.randint(0, n_ent))
                        if nh == si:
                            continue
                        h_neg = self._entity_emb[nh]
                        d_neg = float(np.linalg.norm(h_neg + r - t))
                        # Gradients w.r.t. h_neg, r, t (loss = max(0, margin - d_pos + d_neg))
                        loss = max(0.0, self.margin - d_pos + d_neg)
                        if loss > 0:
                            # gradient direction: increase d_pos, decrease d_neg
                            # d/dh of ||h + r - t|| -> (h + r - t)/||..||
                            # We treat the (h, r, t) and the negative equally simply.
                            # Pre-compute unit vectors:
                            eps = 1e-12
                            g_pos = (h + r - t) / max(d_pos, eps)
                            g_neg = (h_neg + r - t) / max(d_neg, eps)
                            # Update rule (vanilla SGD on the margin loss).
                            # d loss / d h = -g_pos
                            # d loss / d t = +g_pos
                            # d loss / d h_neg = +g_neg
                            # d loss / d r = -g_pos + g_neg
                            self._entity_emb[si] -= lr * (-g_pos)
                            self._entity_emb[oi] -= lr * (+g_pos)
                            self._entity_emb[nh] -= lr * (+g_neg)
                            self._relation_emb[pi] -= lr * (-g_pos + g_neg)
                            total_loss += loss
                    else:
                        # corrupt tail
                        nt = int(self._rng.randint(0, n_ent))
                        if nt == oi:
                            continue
                        t_neg = self._entity_emb[nt]
                        d_neg = float(np.linalg.norm(h + r - t_neg))
                        loss = max(0.0, self.margin - d_pos + d_neg)
                        if loss > 0:
                            eps = 1e-12
                            g_pos = (h + r - t) / max(d_pos, eps)
                            g_neg = (h + r - t_neg) / max(d_neg, eps)
                            self._entity_emb[si] -= lr * (-g_pos)
                            self._entity_emb[oi] -= lr * (+g_pos)
                            self._entity_emb[nt] -= lr * (+g_neg)
                            self._relation_emb[pi] -= lr * (-g_pos + g_neg)
                            total_loss += loss
                    n_examples += 1
            # Renormalise embeddings after each epoch (canonical TransE).
            self._entity_emb = _l2_normalize(self._entity_emb)
            self._relation_emb = _l2_normalize(self._relation_emb)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                avg = total_loss / max(n_examples, 1)
                print(f"[TransE] epoch {epoch + 1}/{epochs} loss={avg:.4f}", flush=True)

        self._trained = True
        self._epochs_run = int(epochs)
        self._lr_used = float(lr)
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def score(self, h: str, r: str, t: str) -> float:
        """Return a margin-based confidence in ``[0, 1]`` for triple ``(h, r, t)``.

        ``score = sigmoid(gamma * (margin - d(h+r, t)))`` so triples whose
        TransE distance is below the margin map to > 0.5 confidence.
        Returns ``0.5`` (a neutral prior) if the embedder is untrained or any
        term is unknown.
        """
        if not self._trained or self._entity_emb is None or self._relation_emb is None:
            return 0.5
        hi = self._entity_to_idx.get(h)
        ri = self._relation_to_idx.get(r)
        ti = self._entity_to_idx.get(t)
        if hi is None or ri is None or ti is None:
            return 0.5
        h_vec = self._entity_emb[hi]
        r_vec = self._relation_emb[ri]
        t_vec = self._entity_emb[ti]
        d = float(np.linalg.norm(h_vec + r_vec - t_vec))
        # margin - d: >0 when the triple is plausible, <0 otherwise.
        raw = self.gamma * (self.margin - d)
        return max(0.0, min(1.0, self._sigmoid(raw)))

    def get_entity_embeddings(self) -> dict[str, np.ndarray]:
        """Return ``{entity_str: embedding_vector}`` for every known entity.

        Returns an empty dict if the embedder hasn't been trained.
        """
        if not self._trained or self._entity_emb is None:
            return {}
        return {self._idx_to_entity[i]: self._entity_emb[i].copy() for i in range(self.n_entities)}

    def get_relation_embeddings(self) -> dict[str, np.ndarray]:
        """Return ``{relation_str: embedding_vector}`` for every known relation."""
        if not self._trained or self._relation_emb is None:
            return {}
        return {
            self._idx_to_relation[i]: self._relation_emb[i].copy() for i in range(self.n_relations)
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))


__all__ = ["TransEEmbedder"]
