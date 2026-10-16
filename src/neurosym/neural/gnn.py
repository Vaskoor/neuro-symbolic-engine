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
"""Graph Neural Network link predictor for the neuro-symbolic engine.

:class:`GNNLinkPredictor` predicts *missing links* (and the confidence /
trust we should place in them) in a :class:`KnowledgeGraph`. It speaks the
same triple vocabulary as the rest of the engine (``s, p, o`` strings) so
the orchestrator can mix its scores with symbolic proofs.

Two backends are supported:

* **PyTorch** (when ``torch`` is installed): a 2-layer Graph Convolutional
  Network (:class:`torch.nn.Module`). Edge weights follow the standard GCN
  normalisation ``D^{-1/2} (A + I) D^{-1/2}``; the scoring head is a
  DistMult-style ``sigmoid((h_s + r_p) . h_o)``.
* **NumPy** (always available, used as the fallback): simple deterministic
  message-passing where each node embedding is the average of its
  neighbours' embeddings (iterated for ``k_layers`` rounds) and the link
  score is a dot product passed through a sigmoid.

Either backend is fully **deterministic** — every PRNG draw flows through
``numpy.random.RandomState(seed)`` (or ``torch.manual_seed(seed)``) so two
training runs over the same triples produce byte-identical predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

try:  # optional: enables the torch.nn.Module 2-layer GCN backend
    import torch
    from torch import nn as torch_nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    torch_nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:  # contract from sibling task ns-A
    from neurosym.core.types import Triple
except ImportError:  # pragma: no cover - fallback
    Triple = None  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover
    from neurosym.core.types import Triple as _Triple


# ---------------------------------------------------------------------------
# Triple coercion helper
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
        return str(getattr(triple, "subject")), str(getattr(triple, "predicate")), str(getattr(triple, "object"))
    raise TypeError(f"Cannot extract (s, p, o) from {type(triple).__name__!r}")


# ---------------------------------------------------------------------------
# Torch backend (2-layer GCN) — only defined when torch is available.
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class _TorchGCN(torch_nn.Module):  # type: ignore[misc]
        """A 2-layer Graph Convolutional Network for link prediction.

        Forward pass:
            H1 = relu(D^{-1/2} (A+I) D^{-1/2} X W1)        # hidden layer
            H2 = D^{-1/2} (A+I) D^{-1/2} H1 W2             # output layer
            score(s, p, o) = sigmoid((H2[s] + R[p]) . H2[o])

        The adjacency normalisation is precomputed as a dense
        ``torch.FloatTensor`` and registered as a buffer.
        """

        def __init__(
            self,
            n_entities: int,
            n_relations: int,
            hidden_dim: int = 32,
            embed_dim: int = 50,
        ) -> None:
            super().__init__()  # type: ignore[no-untyped-call]
            self.n_entities = int(n_entities)
            self.n_relations = int(n_relations)
            self.hidden_dim = int(hidden_dim)
            self.embed_dim = int(embed_dim)
            # Learnable initial node features (one-row-per-entity).
            self.entity_feats = torch_nn.Parameter(
                torch.zeros(self.n_entities, self.embed_dim)
            )
            self.rel_emb = torch_nn.Parameter(
                torch.zeros(max(self.n_relations, 1), self.embed_dim)
            )
            # Two GCN layers.
            self.W1 = torch_nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim))
            self.W2 = torch_nn.Parameter(torch.zeros(self.hidden_dim, self.embed_dim))
            # Adjacency normalisation matrix (filled in by the predictor).
            self.register_buffer(
                "adj_norm",
                torch.eye(self.n_entities),
                persistent=False,
            )
            torch.nn.init.xavier_uniform_(self.entity_feats)  # type: ignore[no-untyped-call]
            torch.nn.init.xavier_uniform_(self.rel_emb)  # type: ignore[no-untyped-call]
            torch.nn.init.xavier_uniform_(self.W1)  # type: ignore[no-untyped-call]
            torch.nn.init.xavier_uniform_(self.W2)  # type: ignore[no-untyped-call]

        def set_adjacency(self, adj_norm: "torch.Tensor") -> None:
            """Replace the normalised adjacency buffer (dense tensor)."""
            self.adj_norm = adj_norm.to(self.W1.device).float()

        def forward(self) -> "torch.Tensor":  # type: ignore[override]
            """Return the (n_entities, embed_dim) node embeddings H2."""
            X = self.entity_feats
            # Layer 1: hidden layer with ReLU
            H1 = torch.relu(self.adj_norm @ X @ self.W1)
            # Layer 2: output layer (no activation — DistMult works on raw)
            H2 = self.adj_norm @ H1 @ self.W2
            return H2

        def score(self, s_idx: "torch.Tensor", p_idx: "torch.Tensor", o_idx: "torch.Tensor") -> "torch.Tensor":
            H = self.forward()
            rel = self.rel_emb[p_idx]
            s_emb = H[s_idx]
            o_emb = H[o_idx]
            return torch.sigmoid(((s_emb + rel) * o_emb).sum(dim=1))


# ---------------------------------------------------------------------------
# GNNLinkPredictor
# ---------------------------------------------------------------------------
_BACKEND_TORCH = "torch"
_BACKEND_NUMPY = "numpy"


def _select_base() -> Any:
    """Return the base class for :class:`GNNLinkPredictor`.

    When ``torch`` is installed the predictor inherits from
    :class:`torch.nn.Module` (so its parameters register and ``.parameters()``
    works); otherwise it inherits from ``object``.
    """
    if _HAS_TORCH:
        return torch_nn.Module  # type: ignore[no-any-return]
    return object


class GNNLinkPredictor(_select_base()):  # type: ignore[misc, valid-type]
    """Predict missing links (and their trust score) in a knowledge graph.

    Parameters
    ----------
    embed_dim:
        Dimensionality of entity/relation embeddings (default 50).
    hidden_dim:
        Width of the GCN's hidden layer (torch backend only).
    seed:
        Seed for the deterministic PRNG (numpy ``RandomState`` and
        ``torch.manual_seed``). Two ``GNNLinkPredictor`` instances built with
        the same seed and trained on the same triples produce identical
        predictions.
    backend:
        Force a specific backend (``"torch"`` or ``"numpy"``). When omitted,
        ``torch`` is used if installed, otherwise numpy.
    """

    def __init__(
        self,
        embed_dim: int = 50,
        hidden_dim: int = 32,
        seed: int = 42,
        backend: str | None = None,
    ) -> None:
        # Initialise the torch.nn.Module base first if applicable.
        if _HAS_TORCH:
            super().__init__()  # type: ignore[no-untyped-call]

        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)

        if backend is None:
            backend = _BACKEND_TORCH if _HAS_TORCH else _BACKEND_NUMPY
        if backend not in (_BACKEND_TORCH, _BACKEND_NUMPY):
            raise ValueError(f"backend must be 'torch' or 'numpy', got {backend!r}")
        if backend == _BACKEND_TORCH and not _HAS_TORCH:
            # torch requested but not installed — fall back.
            backend = _BACKEND_NUMPY
        self.backend: str = backend

        # Vocab maps (populated by ``train``).
        self._entity_to_idx: dict[str, int] = {}
        self._idx_to_entity: list[str] = []
        self._relation_to_idx: dict[str, int] = {}
        self._idx_to_relation: list[str] = []

        # Model state.
        self._entity_emb: np.ndarray | None = None  # (n_entities, embed_dim)
        self._relation_emb: np.ndarray | None = None  # (n_relations, embed_dim)
        self._adj_norm_np: np.ndarray | None = None  # numpy normalised adjacency
        self._torch_model: "_TorchGCN | None" = None
        self._trained: bool = False
        self._epochs_run: int = 0

        # Deterministic PRNGs.
        self._np_rng = np.random.RandomState(self.seed)
        if _HAS_TORCH:
            torch.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def trained(self) -> bool:
        """``True`` after :meth:`train` has been called at least once."""
        return self._trained

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
        # Deterministic, insertion-order sorting for reproducibility.
        ents.sort()
        rels.sort()
        self._idx_to_entity = ents
        self._entity_to_idx = {e: i for i, e in enumerate(ents)}
        self._idx_to_relation = rels
        self._relation_to_idx = {r: i for i, r in enumerate(rels)}

    def _build_adjacency(self, triples: Sequence[tuple[str, str, str]]) -> np.ndarray:
        """Build a dense self-looped, symmetric-normalised adjacency matrix.

        ``A_hat = A + I`` (self-loops), ``D_hat = diag(sum(A_hat))``,
        ``adj_norm = D_hat^{-1/2} A_hat D_hat^{-1/2}``.
        """
        n = self.n_entities
        A = np.zeros((n, n), dtype=np.float64)
        for s, p, o in triples:
            si = self._entity_to_idx[s]
            oi = self._entity_to_idx[o]
            A[si, oi] = 1.0
            A[oi, si] = 1.0  # symmetric (undirected message passing)
        A_hat = A + np.eye(n, dtype=np.float64)
        deg = A_hat.sum(axis=1)
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return adj_norm.astype(np.float64)

    def _init_embeddings(self) -> None:
        n = max(self.n_entities, 1)
        m = max(self.n_relations, 1)
        # Xavier-like init scaled for numpy
        limit = np.sqrt(6.0 / (self.embed_dim + self.embed_dim))
        self._entity_emb = self._np_rng.uniform(
            -limit, limit, size=(n, self.embed_dim)
        ).astype(np.float64)
        self._relation_emb = self._np_rng.uniform(
            -limit, limit, size=(m, self.embed_dim)
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(
        self,
        triples: Iterable[Any],
        epochs: int = 50,
        lr: float = 0.05,
        neg_ratio: int = 1,
        verbose: bool = False,
    ) -> "GNNLinkPredictor":
        """Train the link predictor on a set of positive triples.

        Parameters
        ----------
        triples:
            Iterable of triple-like values (Triple models, ``(s, p, o)``
            tuples, or dicts with ``s``/``p``/``o`` keys).
        epochs:
            Number of full passes over the training triples.
        lr:
            Learning rate for SGD-style updates.
        neg_ratio:
            Number of negative samples drawn per positive triple.
        verbose:
            When True, prints the running loss to stderr at the end of each
            epoch.

        Returns
        -------
        GNNLinkPredictor
            ``self`` (for chaining).
        """
        # Normalise triples to (s, p, o) string tuples.
        spo_list: list[tuple[str, str, str]] = []
        for t in triples:
            try:
                spo_list.append(_triple_as_spo(t))
            except Exception:
                continue
        if not spo_list:
            self._trained = True
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

        self._build_vocab(deduped)
        self._init_embeddings()
        self._adj_norm_np = self._build_adjacency(deduped)

        if self.backend == _BACKEND_TORCH and _HAS_TORCH:
            self._train_torch(deduped, epochs=epochs, lr=lr, neg_ratio=neg_ratio, verbose=verbose)
        else:
            self._train_numpy(deduped, epochs=epochs, lr=lr, neg_ratio=neg_ratio, verbose=verbose)

        self._trained = True
        self._epochs_run = int(epochs)
        return self

    # ------------------------------------------------------------------
    # numpy backend
    # ------------------------------------------------------------------
    def _train_numpy(
        self,
        triples: list[tuple[str, str, str]],
        *,
        epochs: int,
        lr: float,
        neg_ratio: int,
        verbose: bool,
    ) -> None:
        """Simple message-passing link prediction in pure numpy.

        Each epoch:

        1. Compute node embeddings via ``k_layers=2`` rounds of
           ``H = adj_norm @ H_init`` (i.e. average of neighbour embeddings).
        2. Score every positive triple as ``sigmoid((H[s] + R[p]) . H[o])``.
        3. Sample ``neg_ratio`` corrupted tails per positive and score them
           as negatives.
        4. Update ``H_init`` and ``R`` by gradient descent on the BCE loss.
        """
        assert self._entity_emb is not None
        assert self._relation_emb is not None
        assert self._adj_norm_np is not None

        n_ent = self.n_entities
        n_rel = self.n_relations
        pos_idx = [
            (self._entity_to_idx[s], self._relation_to_idx[p], self._entity_to_idx[o])
            for s, p, o in triples
        ]
        # Precompute negative sampling pool
        all_ent_idx = np.arange(n_ent)

        for epoch in range(int(epochs)):
            # Forward: 2-layer message passing (no nonlinearity in numpy path).
            H0 = self._entity_emb
            H1 = self._adj_norm_np @ H0
            H2 = self._adj_norm_np @ H1
            H = H2

            # Gradient accumulator for entity embeddings.
            grad_ent = np.zeros_like(self._entity_emb)
            grad_rel = np.zeros_like(self._relation_emb)

            total_loss = 0.0
            n_examples = 0

            for si, pi, oi in pos_idx:
                h_s = H[si]
                h_o = H[oi]
                r_p = self._relation_emb[pi]
                # Positive score
                pos_score = float(self._sigmoid(float(np.dot(h_s + r_p, h_o))))
                # BCE gradient (w.r.t. score): -(y/score) + (1-y)/(1-score)
                grad_pos_score = -(1.0 - pos_score)
                # chain through sigmoid: d_score/d_pre = s(1-s)
                pre = pos_score * (1.0 - pos_score)
                # grad w.r.t (h_s + r_p) and h_o
                grad_hs_plus_rp = grad_pos_score * pre * h_o
                grad_ho = grad_pos_score * pre * (h_s + r_p)
                grad_ent[si] += grad_hs_plus_rp
                grad_rel[pi] += grad_hs_plus_rp
                grad_ent[oi] += grad_ho
                total_loss += -np.log(max(pos_score, 1e-12))
                n_examples += 1

                # Negatives: corrupt tail.
                for _ in range(neg_ratio):
                    no = int(self._np_rng.randint(0, n_ent))
                    if no == oi:
                        continue
                    h_no = H[no]
                    neg_score = float(self._sigmoid(float(np.dot(h_s + r_p, h_no))))
                    grad_neg_score = neg_score  # d(-log(1-neg))/d_neg_score = neg_score / (1 - neg_score) -> use 0..1 simplification
                    pre_neg = neg_score * (1.0 - neg_score)
                    grad_hs_plus_rp_neg = grad_neg_score * pre_neg * h_no
                    grad_ho_neg = grad_neg_score * pre_neg * (h_s + r_p)
                    grad_ent[si] += grad_hs_plus_rp_neg
                    grad_rel[pi] += grad_hs_plus_rp_neg
                    grad_ent[no] += grad_ho_neg
                    total_loss += -np.log(max(1.0 - neg_score, 1e-12))
                    n_examples += 1

            # Apply gradient update to initial embeddings (and relations).
            # Note: gradients on H propagate back to H_init via the message
            # pass. The simple numpy path approximates this by updating H_init
            # directly (treating the message pass as approximately identity
            # for gradient purposes — sufficient for shallow 2-layer MP).
            self._entity_emb -= lr * grad_ent / max(n_examples, 1)
            self._relation_emb -= lr * grad_rel / max(n_examples, 1)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                avg = total_loss / max(n_examples, 1)
                print(f"[GNN/numpy] epoch {epoch+1}/{epochs} loss={avg:.4f}", flush=True)

    # ------------------------------------------------------------------
    # torch backend
    # ------------------------------------------------------------------
    def _train_torch(
        self,
        triples: list[tuple[str, str, str]],
        *,
        epochs: int,
        lr: float,
        neg_ratio: int,
        verbose: bool,
    ) -> None:
        """2-layer GCN trained with Adam on BCE link-prediction loss."""
        assert _HAS_TORCH
        assert self._adj_norm_np is not None

        n_ent = self.n_entities
        n_rel = self.n_relations
        self._torch_model = _TorchGCN(
            n_entities=n_ent,
            n_relations=n_rel,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        )
        # Push the normalised adjacency into the model.
        adj_t = torch.from_numpy(self._adj_norm_np).float()
        self._torch_model.set_adjacency(adj_t)

        optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=lr)  # type: ignore[no-untyped-call]
        loss_fn = torch_nn.BCELoss()  # type: ignore[no-untyped-call]

        pos_idx = [
            (self._entity_to_idx[s], self._relation_to_idx[p], self._entity_to_idx[o])
            for s, p, o in triples
        ]
        s_pos = torch.tensor([t[0] for t in pos_idx], dtype=torch.long)
        p_pos = torch.tensor([t[1] for t in pos_idx], dtype=torch.long)
        o_pos = torch.tensor([t[2] for t in pos_idx], dtype=torch.long)
        ones = torch.ones(len(pos_idx), dtype=torch.float32)

        n_ent_t = int(n_ent)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()  # type: ignore[no-untyped-call]
            pos_scores = self._torch_model.score(s_pos, p_pos, o_pos)
            pos_loss = loss_fn(pos_scores, ones)
            # Negatives: corrupt tails.
            if neg_ratio > 0 and len(pos_idx) > 0:
                neg_o = torch.randint(0, n_ent_t, (len(pos_idx) * neg_ratio,), dtype=torch.long)
                neg_s = s_pos.repeat_interleave(neg_ratio)
                neg_p = p_pos.repeat_interleave(neg_ratio)
                neg_scores = self._torch_model.score(neg_s, neg_p, neg_o)
                neg_targets = torch.zeros_like(neg_scores)
                neg_loss = loss_fn(neg_scores, neg_targets)
                loss = pos_loss + neg_loss
            else:
                loss = pos_loss
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()  # type: ignore[no-untyped-call]
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"[GNN/torch] epoch {epoch+1}/{epochs} loss={float(loss):.4f}", flush=True)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict_link(self, s: str, p: str, o: str) -> float:
        """Return a confidence score in ``[0, 1]`` for the triple ``(s, p, o)``.

        If the predictor hasn't been trained, returns a deterministic
        hash-based prior so callers always get a number in range. If any of
        ``s``/``p``/``o`` is unknown at inference time, falls back to a
        deterministic hash-based score.
        """
        if not self._trained or self._entity_emb is None or self._relation_emb is None:
            return self._hash_prior(s, p, o)
        si = self._entity_to_idx.get(s)
        pi = self._relation_to_idx.get(p)
        oi = self._entity_to_idx.get(o)
        if si is None or pi is None or oi is None:
            return self._hash_prior(s, p, o)

        if self.backend == _BACKEND_TORCH and _HAS_TORCH and self._torch_model is not None:
            with torch.no_grad():
                si_t = torch.tensor([si], dtype=torch.long)
                pi_t = torch.tensor([pi], dtype=torch.long)
                oi_t = torch.tensor([oi], dtype=torch.long)
                score = float(self._torch_model.score(si_t, pi_t, oi_t)[0])
            return max(0.0, min(1.0, score))

        # numpy path: recompute embeddings via 2-layer message pass.
        assert self._adj_norm_np is not None
        H0 = self._entity_emb
        H1 = self._adj_norm_np @ H0
        H2 = self._adj_norm_np @ H1
        h_s = H2[si]
        h_o = H2[oi]
        r_p = self._relation_emb[pi]
        raw = float(np.dot(h_s + r_p, h_o))
        score = float(self._sigmoid(raw))
        return max(0.0, min(1.0, score))

    def get_embeddings(self) -> dict[str, np.ndarray]:
        """Return a dict mapping each entity string to its embedding vector.

        The embeddings are the post-message-passing representations (after
        the 2-layer GCN forward pass), so they already encode graph context.
        Returns an empty dict if the model hasn't been trained.
        """
        if not self._trained or self._entity_emb is None:
            return {}
        if self.backend == _BACKEND_TORCH and _HAS_TORCH and self._torch_model is not None:
            with torch.no_grad():
                H = self._torch_model.forward().cpu().numpy()
            return {self._idx_to_entity[i]: H[i] for i in range(self.n_entities)}
        assert self._adj_norm_np is not None
        H0 = self._entity_emb
        H1 = self._adj_norm_np @ H0
        H2 = self._adj_norm_np @ H1
        return {self._idx_to_entity[i]: H2[i].copy() for i in range(self.n_entities)}

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

    def _hash_prior(self, s: str, p: str, o: str) -> float:
        """Deterministic hash-based prior for unseen triples.

        Returns a value in ``[0.05, 0.5]`` — a low-confidence, neutral
        default so unknown triples never fool downstream logic.
        """
        h = hash((s, p, o, self.seed)) & 0xFFFFFFFF
        # map [0, 2^32) -> [0.05, 0.5]
        scaled = h / 0xFFFFFFFF
        return float(0.05 + 0.45 * scaled)


__all__ = ["GNNLinkPredictor"]
