# SPDX-License-Identifier: MIT
#
# neurosym.core.config
# ---------------------
# YAML-driven configuration for the Neuro-Symbolic Engine.
#
# Only ``pyyaml`` is required to load a config file; everything else is an
# optional dependency resolved lazily by the subsystems that need it.

"""Engine configuration objects and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class KGConfig(BaseModel):
    """Knowledge graph wiring."""

    turtle_files: list[str] = Field(
        default_factory=list, description="Turtle files loaded into the KG at start-up."
    )
    base_namespace: str = Field(
        default="http://example.org/sci#",
        description="Default namespace bound to the 'ex' prefix.",
    )


class DatalogConfig(BaseModel):
    """Symbolic (Datalog) reasoner wiring."""

    rules_files: list[str] = Field(
        default_factory=list, description="YAML files of Datalog rules to load."
    )
    max_iterations: int = Field(
        default=100, ge=1, description="Forward-chaining fixpoint iteration cap."
    )
    seed: int = Field(default=42, description="Determinism seed for tie-breaking.")


class SHACLConfig(BaseModel):
    """SHACL validation wiring."""

    enabled: bool = Field(default=True, description="Whether to expose SHACL validation.")
    shapes_files: list[str] = Field(
        default_factory=list,
        description="Turtle SHACL shapes graphs (empty => validate shapes inside the KG).",
    )


class NeuralConfig(BaseModel):
    """Neural backend wiring (optional; guarded)."""

    enabled: bool = Field(default=False)
    model: str = Field(default="", description="Model name / path.")
    weight_path: Optional[str] = Field(default=None)
    device: str = Field(default="cpu")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """LLM backend wiring (optional; guarded)."""

    enabled: bool = Field(default=False)
    provider: str = Field(default="", description="e.g. 'openai', 'local'.")
    model: str = Field(default="")
    api_key_env: str = Field(default="", description="Env var holding the API key.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EngineConfig(BaseModel):
    """Top-level engine configuration."""

    name: str = Field(default="neurosym")
    version: str = Field(default="0.1.0")
    seed: int = Field(default=42, description="Master seed for deterministic behaviour.")
    kg: KGConfig = Field(default_factory=KGConfig)
    datalog: DatalogConfig = Field(default_factory=DatalogConfig)
    shacl: SHACLConfig = Field(default_factory=SHACLConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cascade: list[str] = Field(
        default_factory=lambda: ["kg", "symbolic", "neural", "llm"],
        description="Ordered fallback chain of source labels.",
    )
    logging: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> EngineConfig:
    """Load a YAML config file and validate it into :class:`EngineConfig`.

    Missing keys fall back to the pydantic defaults, so a minimal config
    can be as small as ``name: neurosym``.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return EngineConfig.model_validate(data)


__all__ = [
    "KGConfig",
    "DatalogConfig",
    "SHACLConfig",
    "NeuralConfig",
    "LLMConfig",
    "EngineConfig",
    "load_config",
]
