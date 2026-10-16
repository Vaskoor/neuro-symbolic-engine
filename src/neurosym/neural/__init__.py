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
"""Neural layer of the neuro-symbolic engine.

Public surface:

* :class:`GNNLinkPredictor`  - 2-layer GCN (torch) or numpy message-passing
  fallback for missing-link prediction.
* :class:`TransEEmbedder`    - TransE (h + r ≈ t) embeddings trained in pure
  numpy via SGD on the margin-ranking loss.

Both components are fully deterministic — seeded PRNGs and no I/O.
"""

from __future__ import annotations  # noqa: F401

from .embeddings import TransEEmbedder  # noqa: F401
from .gnn import GNNLinkPredictor  # noqa: F401

__all__ = ["GNNLinkPredictor", "TransEEmbedder"]

__version__ = "0.1.0"
