# SPDX-License-Identifier: MIT
"""neurosym.core — shared contract, configuration and orchestrating engine.

Importing this package eagerly wires the canonical types and configuration
helpers. The :class:`NeuroSymbolicEngine` is imported last because it depends
on the :mod:`neurosym.knowledge` subsystem.
"""

from __future__ import annotations  # noqa: F401

from .config import EngineConfig, load_config  # noqa: F401

# The engine depends on the knowledge subsystem; import it last so that the
# contract objects above remain available even if a heavyweight optional
# dependency were missing.
from .engine import NeuroSymbolicEngine  # noqa: F401
from .types import (  # noqa: F401
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
