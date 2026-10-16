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
"""LLM bridge for the neuro-symbolic engine.

Public surface:

* :class:`LLMBridge`  - NL→SPARQL, candidate-fact generation and
  proof-to-NL explanation. Uses ``z-ai-web-dev-sdk`` when available
  (guarded); falls back to deterministic, seeded NL templates otherwise.
"""

from __future__ import annotations  # noqa: F401

from .bridge import LLMBridge  # noqa: F401

__all__ = ["LLMBridge"]

__version__ = "0.1.0"
