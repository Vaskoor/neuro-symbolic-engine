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
"""Symbolic reasoning package for the Neuro-Symbolic Engine.

Public surface:
    * :class:`SymbolicReasoner`     - Datalog forward-chaining + OWL subsumption.
    * :class:`ProofBuilder`        - Builds :class:`ProofTrace` from firing logs.
    * :func:`format_proof`          - Human-readable proof tree rendering.
    * :class:`QueryEngine`          - NL/SPARQL/Datalog pattern translation.

The submodules import types from :mod:`neurosym.core.types` and engine classes
from :mod:`neurosym.knowledge.graph` / :mod:`neurosym.knowledge.datalog`. Those
contracts are authored by the parallel ``ns-A`` task. When those modules are not
yet present at import time, lightweight fallback aliases (pydantic models with
the same shape) are defined in each submodule so the package remains importable
in isolation.
"""

from __future__ import annotations

from .proof import ProofBuilder, format_proof
from .queries import QueryEngine
from .reasoner import SymbolicReasoner

__all__ = [
    "SymbolicReasoner",
    "ProofBuilder",
    "format_proof",
    "QueryEngine",
]

__version__ = "0.1.0"
