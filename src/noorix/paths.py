"""Helpers for resolving and safely reading paths stored in local configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any

MAX_LOCAL_JSON_BYTES = 1024 * 1024


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve relative values beside a config file without hiding linked parents."""
    config_candidate = Path(config_path).expanduser()
    if _has_symlink_component(config_candidate):
        raise ValueError("Configuration path contains a symbolic link or junction")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return config_candidate.resolve().parent / candidate


def _has_symlink_component(path: Path) -> bool:
    """Check the candidate and existing ancestors without following links."""
    candidate = Path(path).absolute()
    for component in (candidate, *candidate.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return True
        # FILE_ATTRIBUTE_REPARSE_POINT also detects Windows junctions on Python 3.11,
        # where Path.is_junction() is not available.
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            return True
        is_junction = getattr(component, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
    return False


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every JSON object level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON document contains a duplicate object key")
        result[key] = value
    return result


def load_json_document(path: str | Path, *, max_bytes: int = MAX_LOCAL_JSON_BYTES) -> Any:
    """Read a bounded regular JSON file, rejecting symlink paths and duplicate keys.

    The descriptor and path metadata are compared before and after the bounded
    read. This detects observable replacement or mutation but cannot eliminate
    every concurrent filesystem race.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    candidate = Path(path).expanduser()
    if _has_symlink_component(candidate):
        raise ValueError("JSON path contains a symbolic link or junction")
    before_path = candidate.lstat()
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError("JSON path is not a regular file")
    with candidate.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(before_path):
            raise ValueError("JSON file changed before it could be read")
        raw = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    after_path = candidate.lstat()
    if _file_identity(before) != _file_identity(after) or _file_identity(after) != _file_identity(after_path):
        raise ValueError("JSON file changed while it was being read")
    if len(raw) > max_bytes:
        raise ValueError(f"JSON document exceeds the {max_bytes}-byte read limit")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except RecursionError:
        raise ValueError("JSON document nesting exceeds the parser limit") from None


def require_regular_file(path: str | Path, *, label: str = "File") -> Path:
    """Require an existing regular file with no symlink or junction ancestors."""
    candidate = Path(path).expanduser()
    if _has_symlink_component(candidate):
        raise ValueError(f"{label} path contains a symbolic link or junction")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} path must be a regular file")
    return candidate
