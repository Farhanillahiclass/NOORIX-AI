"""Verify model artifact SHA-256 manifests without loading model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


def verify_manifest(root: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Verify every manifest entry without following symlinks or escaping root."""
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    try:
        document: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [{"path": str(manifest_path), "status": "ERROR", "detail": f"manifest unreadable: {exc}"}]

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
        pure = PurePosixPath(normalized)
        key = pure.as_posix().casefold()
        if pure.is_absolute() or ".." in pure.parts or ":" in normalized or key in seen:
            results.append({"path": relative, "status": "ERROR", "detail": "unsafe or duplicate manifest path"})
            continue
        seen.add(key)
        candidate = root.joinpath(*pure.parts)
        try:
            # Reject symlinks in every component, including parent directories.
            # Checking only candidate.is_symlink() misses an in-root symlinked
            # directory that redirects traversal to another file.
            current = root
            has_symlink_component = False
            for part in pure.parts:
                current = current / part
                if current.is_symlink():
                    has_symlink_component = True
                    break
            if (
                has_symlink_component
                or not candidate.is_file()
                or not candidate.resolve().is_relative_to(root)
            ):
                results.append({"path": relative, "status": "ERROR", "detail": "missing, non-regular, symlink, or outside model root"})
                continue
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        except OSError as exc:
            results.append({"path": relative, "status": "ERROR", "detail": f"read failed: {exc}"})
            continue
        if actual.lower() != expected.lower():
            results.append({"path": relative, "status": "MISMATCH", "detail": "SHA-256 does not match"})
        else:
            results.append({"path": relative, "status": "PASS", "detail": "SHA-256 verified"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHA-256 checksums for model files.")
    parser.add_argument("--root", type=Path, required=True, help="Trusted root directory containing model artifacts.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest containing SHA-256 entries.")
    args = parser.parse_args()
    results = verify_manifest(args.root, args.manifest)
    for item in results:
        print(f"[{item['status']}] {item['path']}: {item['detail']}")
    failures = sum(item["status"] != "PASS" for item in results)
    print(f"Verified: {len(results) - failures}; problems: {failures}")
    return 0 if results and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
