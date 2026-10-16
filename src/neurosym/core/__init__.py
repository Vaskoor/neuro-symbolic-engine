# SPDX-License-Identifier: MIT
"""neurosym.core — shared contract, configuration and orchestrating engine.

Importing this package eagerly wires the canonical types and configuration
helpers. The :class:`NeuroSymbolicEngine` is imported last because it depends
on the :mod:`neurosym.knowledge` subsystem.
"""

from __future__ import annotations

from .config import EngineConfig, load_config

# The engine depends on the knowledge subsystem; import it last so that the
# contract objects above remain available even if a heavyweight optional
# dependency were missing.
from .engine import NeuroSymbolicEngine
from .types import (
    ProofStep,
    ProofTrace,
    Query,
    QueryType,
    ReasoningResult,
    Rule,
    SourceLabel,
    Triple,
    Violation,
)

__all__ = [
    "EngineConfig",
    "load_config",
    "NeuroSymbolicEngine",
    "Triple",
    "Rule",
    "ProofStep",
    "ProofTrace",
    "ReasoningResult",
    "Query",
    "QueryType",
    "Violation",
    "SourceLabel",
]
