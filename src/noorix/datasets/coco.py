"""Dependency-free structural validation for COCO-style detection annotations."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_coco_dataset(
    annotation_file: str | Path,
    images_dir: str | Path | None = None,
) -> list[str]:
    """Return validation errors; an empty list means structural checks passed.

    If images_dir is supplied, every image path must resolve to an existing
    file inside that directory. No images are opened or modified.
    """
    annotation_path = Path(annotation_file)
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"Cannot read annotation file: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]

    if not isinstance(payload, dict):
        return ["Root JSON value must be an object."]

    errors: list[str] = []
    for key in ("images", "categories", "annotations"):
        if not isinstance(payload.get(key), list):
            errors.append(f"'{key}' must be an array.")
    if errors:
        return errors

    images = payload["images"]
    categories = payload["categories"]
    annotations = payload["annotations"]

    image_ids: set[int] = set()
    category_ids: set[int] = set()
    annotation_ids: set[int] = set()
    image_dimensions: dict[int, tuple[int, int]] = {}
    image_root = Path(images_dir).resolve() if images_dir is not None else None

    for index, image in enumerate(images):
        prefix = f"images[{index}]"
        if not isinstance(image, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        image_id = image.get("id")
        if not _is_int(image_id):
            errors.append(f"{prefix}.id must be an integer.")
        elif image_id in image_ids:
            errors.append(f"{prefix}.id duplicates {image_id}.")
        else:
            image_ids.add(image_id)

        filename = image.get("file_name")
        safe_filename = isinstance(filename, str) and bool(filename.strip())
        if safe_filename:
            relative = PurePosixPath(filename)
            safe_filename = not relative.is_absolute() and ".." not in relative.parts and "\\" not in filename
        if not safe_filename:
            errors.append(f"{prefix}.file_name must be a safe, non-empty relative POSIX path.")
        elif image_root is not None:
            candidate = (image_root / Path(*PurePosixPath(filename).parts)).resolve()
            if not candidate.is_relative_to(image_root):
                errors.append(f"{prefix}.file_name resolves outside the image directory.")
            elif not candidate.is_file():
                errors.append(f"{prefix}.file_name does not exist: {filename}")

        width, height = image.get("width"), image.get("height")
        valid_dimensions = _is_int(width) and width > 0 and _is_int(height) and height > 0
        if not valid_dimensions:
            errors.append(f"{prefix}.width and height must be positive integers.")
        elif _is_int(image_id):
            image_dimensions[image_id] = (width, height)

    for index, category in enumerate(categories):
        prefix = f"categories[{index}]"
        if not isinstance(category, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        category_id = category.get("id")
        if not _is_int(category_id):
            errors.append(f"{prefix}.id must be an integer.")
        elif category_id in category_ids:
            errors.append(f"{prefix}.id duplicates {category_id}.")
        else:
            category_ids.add(category_id)
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            errors.append(f"{prefix}.name must be a non-empty string.")

    for index, annotation in enumerate(annotations):
        prefix = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        annotation_id = annotation.get("id")
        if not _is_int(annotation_id):
            errors.append(f"{prefix}.id must be an integer.")
        elif annotation_id in annotation_ids:
            errors.append(f"{prefix}.id duplicates {annotation_id}.")
        else:
            annotation_ids.add(annotation_id)

        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if not _is_int(image_id) or image_id not in image_ids:
            errors.append(f"{prefix}.image_id does not reference a known image.")
        if not _is_int(category_id) or category_id not in category_ids:
            errors.append(f"{prefix}.category_id does not reference a known category.")

        bbox = annotation.get("bbox")
        valid_bbox = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(_is_finite_number(value) for value in bbox)
        )
        if not valid_bbox:
            errors.append(f"{prefix}.bbox must contain four finite numbers [x, y, width, height].")
            continue

        x, y, box_width, box_height = bbox
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
            errors.append(f"{prefix}.bbox must have non-negative origin and positive dimensions.")
            continue

        dimensions = image_dimensions.get(image_id) if _is_int(image_id) else None
        if dimensions is not None:
            image_width, image_height = dimensions
            if x + box_width > image_width or y + box_height > image_height:
                errors.append(f"{prefix}.bbox extends beyond image {image_id} bounds.")

        area = annotation.get("area")
        if not _is_finite_number(area) or area <= 0:
            errors.append(f"{prefix}.area must be a positive finite number.")

        iscrowd = annotation.get("iscrowd", 0)
        if not _is_int(iscrowd) or iscrowd not in (0, 1):
            errors.append(f"{prefix}.iscrowd must be 0 or 1.")

    return errors
