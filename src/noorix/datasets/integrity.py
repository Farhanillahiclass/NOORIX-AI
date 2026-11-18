"""Read-only image integrity and exact-duplicate analysis for COCO splits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any


def _safe_image_path(filename: Any) -> bool:
    if not isinstance(filename, str) or not filename.strip():
        return False
    relative = PurePosixPath(filename)
    return not relative.is_absolute() and ".." not in relative.parts and "\\" not in filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_coco_integrity(splits: dict[str, dict[str, Any]], *, verify_image_decode: bool = False) -> dict[str, Any]:
    """Inspect referenced files and group byte-identical images by SHA-256."""
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a non-empty mapping.")
    if verify_image_decode:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for decode verification; install the camera extra.") from exc
    else:
        Image = None  # noqa: N816

    hash_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing, unreadable, invalid_images, unsafe_paths = [], [], [], []
    checked = total = 0
    for split_name in sorted(splits):
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("split names must be non-empty strings.")
        entry = splits[split_name]
        if not isinstance(entry, dict):
            raise ValueError(f"split '{split_name}' must be an object.")
        payload, images_dir = entry.get("payload"), entry.get("images_dir")
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            raise ValueError(f"split '{split_name}' payload must contain an images array.")
        if images_dir is None:
            raise ValueError(f"split '{split_name}' requires images_dir.")
        root = Path(images_dir).resolve()
        for image in payload["images"]:
            total += 1
            filename = image.get("file_name") if isinstance(image, dict) else None
            item = {"split": split_name, "file_name": str(filename or "")}
            if not isinstance(filename, str) or not filename.strip():
                unsafe_paths.append({**item, "reason": "missing or empty file_name"})
                continue
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in filename:
                unsafe_paths.append({**item, "reason": "not a safe relative POSIX path"})
                continue
            candidate = (root / Path(*relative.parts)).resolve()
            if not candidate.is_relative_to(root):
                unsafe_paths.append({**item, "reason": "resolved outside images_dir"})
                continue
            if not candidate.is_file():
                missing.append(item)
                continue
            try:
                digest = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hexdigest = digest.hexdigest()
            except OSError as exc:
                unreadable.append({**item, "reason": str(exc)})
                continue
            checked += 1
            hash_locations[hexdigest].append(item)
            if verify_image_decode:
                try:
                    with Image.open(candidate) as image_file:
                        image_file.verify()
                except Exception as exc:
                    invalid_images.append({**item, "reason": str(exc)})

    duplicate_groups = [
        {"sha256": digest, "file_count": len(locations), "locations": locations}
        for digest, locations in sorted(hash_locations.items()) if len(locations) > 1
    ]
    return {
        "schema_version": 1,
        "summary": {
            "splits": len(splits), "referenced_images": total, "files_hashed": checked,
            "missing_files": len(missing), "unreadable_files": len(unreadable),
            "unsafe_paths": len(unsafe_paths), "decode_failures": len(invalid_images),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_file_entries": sum(group["file_count"] for group in duplicate_groups),
            "decode_verification": "enabled" if verify_image_decode else "not_requested",
        },
        "duplicate_groups": duplicate_groups, "missing_files": missing,
        "unreadable_files": unreadable, "unsafe_paths": unsafe_paths,
        "decode_failures": invalid_images,
        "limitations": [
            "SHA-256 detects byte-identical files only; resized or recompressed copies are not detected.",
            "Decode verification is optional and requires Pillow; success does not certify labels or suitability.",
            "Files are read only and are never changed, moved or deleted.",
        ],
    }
