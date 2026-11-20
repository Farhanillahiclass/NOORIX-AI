"""Verify model artifact SHA-256 manifests without loading model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


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


def verify_manifest(root: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Verify every manifest entry without following symlinks or escaping root."""
    root = root.resolve()
    if not _manifest_location_is_safe(root, manifest_path):
        return [{"path": str(manifest_path), "status": "ERROR",
                 "detail": "manifest must be a regular non-symlink file inside the trusted model root"}]
    manifest_path = manifest_path.absolute()
    try:
        document: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest unreadable"}]

    if not isinstance(document, dict):
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest root must be an object"}]
    integrity = document.get("integrity", document)
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest algorithm must be sha256"}]
    entries = integrity.get("files")
    if not isinstance(entries, list) or not entries:
        return [{"path": str(manifest_path), "status": "ERROR", "detail": "manifest files must be a non-empty list"}]

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            results.append({"path": f"entry[{index}]", "status": "ERROR", "detail": "entry must be an object"})
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not relative or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            results.append({"path": str(relative), "status": "ERROR", "detail": "path and 64-character SHA-256 are required"})
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
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
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
