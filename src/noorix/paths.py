"""Helpers for resolving paths stored in a local configuration file."""
from __future__ import annotations

from pathlib import Path


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve relative values beside the config file, not the process CWD."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(config_path).expanduser().resolve().parent / candidate
