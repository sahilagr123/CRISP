"""YAML config loader for CRISP."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from crisp.config import (
    AdvantageConfig,
    CoachConfig,
    CRISPConfig,
    GRPOConfig,
    InfraConfig,
    PlayerConfig,
    TrainingConfig,
)

_SECTION_MAP = {
    "player": PlayerConfig,
    "coach": CoachConfig,
    "advantage": AdvantageConfig,
    "grpo": GRPOConfig,
    "infra": InfraConfig,
    "training": TrainingConfig,
}


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> CRISPConfig:
    """Load a CRISPConfig from a YAML file with optional dot-path overrides.

    Overrides use dot-notation: ``{"infra.learning_rate": 1e-4}``
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for dotpath, value in overrides.items():
            _apply_override(raw, dotpath, value)

    sections = {}
    for key, cls in _SECTION_MAP.items():
        if key in raw:
            sections[key] = cls(**raw[key])

    return CRISPConfig(**sections)


def _apply_override(raw: dict, dotpath: str, value: Any) -> None:
    """Apply a single dot-path override to the raw dict."""
    parts = dotpath.split(".")
    d = raw
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
