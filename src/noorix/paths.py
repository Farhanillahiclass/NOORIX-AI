"""Helpers for resolving paths stored in a local configuration file."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve absolute paths, config-relative paths, then optional model-root paths.

    Set NOORIX_MODELS_DIR once to point at an extracted offline model package.
    Existing config-relative files always take precedence.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    config_candidate = Path(config_path).expanduser().resolve().parent / candidate
    if config_candidate.exists():
        return config_candidate
    model_root = os.environ.get("NOORIX_MODELS_DIR")
    if model_root:
        root_candidate = Path(model_root).expanduser() / candidate
        if root_candidate.exists():
            return root_candidate
    return config_candidate
