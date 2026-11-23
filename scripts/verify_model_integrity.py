"""Verify model artifact SHA-256 manifests without loading model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import os
import stat
from typing import Any

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_ENTRIES = 20_000
MAX_MANIFEST_PATH_LENGTH = 4_096


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _has_symlink_component(path: Path) -> bool:
    """Detect a symlink at the path or any existing ancestor."""
    candidate = Path(path).absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _trusted_root_is_safe(root: Path) -> bool:
    """Require the supplied model root to be an existing non-symlink directory."""
    try:
        return not _has_symlink_component(root) and Path(root).is_dir()
    except (OSError, RuntimeError):
        return False


def _manifest_location_is_safe(root: Path, manifest_path: Path) -> bool:
    """Require the manifest itself to be a regular, non-symlink file under root."""
    try:
        root = root.resolve()
        supplied = manifest_path.absolute()
        relative = supplied.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
        return current.is_file() and current.resolve().is_relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False


def _hash_file_stably(path: Path) -> str | None:
    """Hash one regular file and reject observable changes during the read."""
    try:
        before_path = path.lstat()
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            return None
        digest = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            identity_before = (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            path_identity_before = (
                before_path.st_dev, before_path.st_ino, before_path.st_size,
                before_path.st_mtime_ns, before_path.st_ctime_ns,
            )
            if identity_before != path_identity_before:
                return None
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        after_path = path.lstat()
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        path_identity_after = (
            after_path.st_dev, after_path.st_ino, after_path.st_size,
            after_path.st_mtime_ns, after_path.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_after != path_identity_after
            or bytes_read != before.st_size
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(after_path.st_mode)
        ):
            return None
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def verify_manifest(root: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Verify every manifest entry without following symlinks or escaping root."""
    supplied_root = Path(root)
    if not _trusted_root_is_safe(supplied_root):
        return [{"path": str(manifest_path), "status": "ERROR",
                 "detail": "trusted model root must be an existing non-symlink directory"}]
    root = supplied_root.resolve()
    if not _manifest_location_is_safe(root, manifest_path):
        return [{"path": str(manifest_path), "status": "ERROR",
                 "detail": "manifest must be a regular non-symlink file inside the trusted model root"}]
    manifest_path = manifest_path.absolute()
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            return [{"path": str(manifest_path), "status": "ERROR",
                     "detail": f"manifest exceeds the {MAX_MANIFEST_BYTES}-byte size limit"}]
        with manifest_path.open("rb") as stream:
            raw_manifest = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(raw_manifest) > MAX_MANIFEST_BYTES:
            return [{"path": str(manifest_path), "status": "ERROR",
                     "detail": f"manifest exceeds the {MAX_MANIFEST_BYTES}-byte size limit"}]
        document: Any = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, ValueError):
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest unreadable"}]

    if not isinstance(document, dict):
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest root must be an object"}]
    if "integrity" in document and ("algorithm" in document or "files" in document):
        return [{"path": str(manifest_path), "status": "ERROR",
                 "detail": "manifest must use either nested integrity or flat checksum fields, not both"}]
    integrity = document.get("integrity", document)
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest algorithm must be sha256"}]
    entries = integrity.get("files")
    if not isinstance(entries, list) or not entries:
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest files must be a non-empty list"}]
    if len(entries) > MAX_MANIFEST_ENTRIES:
        return [{"path": str(manifest_path), "status": "ERROR",
                 "detail": f"manifest exceeds the {MAX_MANIFEST_ENTRIES}-entry limit"}]

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            results.append({"path": f"entry[{index}]", "status": "ERROR", "detail": "entry must be an object"})
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if (not isinstance(relative, str) or not relative
                or len(relative) > MAX_MANIFEST_PATH_LENGTH
                or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)):
            results.append({"path": str(relative)[:MAX_MANIFEST_PATH_LENGTH],
                            "status": "ERROR",
                            "detail": "a bounded relative path and 64-character SHA-256 are required"})
            continue
        normalized = relative.replace("\\", "/")
        raw_parts = normalized.split("/")
        windows_reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{number}" for number in range(1, 10)),
            *(f"LPT{number}" for number in range(1, 10)),
            "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
        }
        has_unsafe_component = any(
            not part
            or part in {".", ".."}
            or part.endswith((" ", "."))
            or "\x00" in part
            or part.split(".", 1)[0].rstrip(" .").upper() in windows_reserved
            for part in raw_parts
        )
        pure = PurePosixPath(normalized)
        key = pure.as_posix().casefold()
        if pure.is_absolute() or has_unsafe_component or ":" in normalized or key in seen:
            results.append({"path": relative, "status": "ERROR", "detail": "unsafe or duplicate manifest path"})
            continue
        seen.add(key)
        candidate = root.joinpath(*pure.parts)
        try:
            current = root
            has_symlink_component = False
            for part in pure.parts:
                current = current / part
                if current.is_symlink():
                    has_symlink_component = True
                    break
            if has_symlink_component or not candidate.is_file() or not candidate.resolve().is_relative_to(root):
                results.append({"path": relative, "status": "ERROR", "detail": "missing, non-regular, symlink, or outside model root"})
                continue
            actual = _hash_file_stably(candidate)
            if actual is None:
                results.append({
                    "path": relative,
                    "status": "ERROR",
                    "detail": "model file changed during verification or could not be read",
                })
                continue
        except OSError:
            results.append({"path": relative, "status": "ERROR", "detail": "model file could not be read"})
            continue
        if actual.lower() != expected.lower():
            results.append({"path": relative, "status": "MISMATCH", "detail": "SHA-256 does not match"})
        else:
            results.append({"path": relative, "status": "PASS", "detail": "SHA-256 verified"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHA-256 checksums for model files.")
    parser.add_argument("--root", type=Path, required=True, help="Trusted root directory containing model artifacts.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest inside the trusted model root.")
    args = parser.parse_args()
    results = verify_manifest(args.root, args.manifest)
    for item in results:
        print(f"[{item['status']}] {item['path']}: {item['detail']}")
    failures = sum(item["status"] != "PASS" for item in results)
    print(f"Verified: {len(results) - failures}; problems: {failures}")
    return 0 if results and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
