"""Metadata-only adapters for supported object-detection annotation formats.

These functions never download, open, copy, or redistribute image assets.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


def _require_list(payload: Any, key: str) -> list:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"Expected an object containing a '{key}' array.")
    return payload[key]


def _normalise_coco_like(
    payload: dict[str, Any],
    *,
    source_dataset: str,
    source_format: str,
    category_id_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    _require_list(payload, "images")
    _require_list(payload, "categories")
    _require_list(payload, "annotations")
    mapping = dict(category_id_map or {})
    result = deepcopy(payload)

    if mapping:
        for category in result["categories"]:
            old_id = category.get("id")
            if old_id in mapping:
                category["id"] = mapping[old_id]
        for annotation in result["annotations"]:
            old_id = annotation.get("category_id")
            if old_id in mapping:
                annotation["category_id"] = mapping[old_id]

    result.setdefault("info", {})
    if not isinstance(result["info"], dict):
        raise ValueError("The COCO 'info' field must be an object when present.")
    result["noorix_source"] = {
        "dataset": source_dataset,
        "format": source_format,
        "conversion": "metadata-only",
        "source_category_ids_preserved": not bool(mapping),
    }
    # Keep source-specific image/annotation/category fields intact. In particular,
    # LVIS federated fields such as neg_category_ids and not_exhaustive_category_ids
    # must not be discarded by a generic COCO normalisation step.
    return result


def adapt_coco(
    payload: dict[str, Any],
    *,
    source_dataset: str = "coco",
    category_id_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Validate the basic COCO collections and return a loss-minimising copy."""
    return _normalise_coco_like(
        payload,
        source_dataset=source_dataset,
        source_format="coco",
        category_id_map=category_id_map,
    )


def adapt_lvis(
    payload: dict[str, Any],
    *,
    category_id_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Adapt LVIS' COCO-like JSON while retaining federated/long-tail fields."""
    return _normalise_coco_like(
        payload,
        source_dataset="lvis",
        source_format="lvis",
        category_id_map=category_id_map,
    )


def adapt_objects365(
    payload: dict[str, Any],
    *,
    category_id_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Adapt an Objects365 annotation export that already uses COCO collections.

    Objects365 releases/exports can differ. This adapter deliberately accepts
    only the explicit images/categories/annotations shape; it does not guess
    undocumented source schemas or invent category mappings.
    """
    return _normalise_coco_like(
        payload,
        source_dataset="objects365",
        source_format="objects365-coco-export",
        category_id_map=category_id_map,
    )


def adapt_bdd100k(
    frames: list[dict[str, Any]],
    *,
    image_dimensions: Mapping[str, tuple[int, int]],
    source_dataset: str = "bdd100k",
) -> dict[str, Any]:
    """Convert BDD100K detection frame labels to COCO-style metadata.

    image_dimensions must map each frame's name to (width, height), obtained
    from a separately verified image inventory. No image files are read.
    """
    if not isinstance(frames, list):
        raise ValueError("BDD100K labels must be a JSON array of frame objects.")

    category_names: set[str] = set()
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{frame_index}] must be an object.")
        labels = frame.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"frames[{frame_index}].labels must be an array.")
        for label_index, label in enumerate(labels):
            if not isinstance(label, dict):
                raise ValueError(f"frames[{frame_index}].labels[{label_index}] must be an object.")
            category = label.get("category")
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"frames[{frame_index}].labels[{label_index}].category is required.")
            if label.get("box2d") is not None:
                category_names.add(category)

    categories = [
        {"id": index + 1, "name": name, "source_category_name": name}
        for index, name in enumerate(sorted(category_names))
    ]
    category_ids = {item["name"]: item["id"] for item in categories}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for image_id, frame in enumerate(frames, start=1):
        file_name = frame.get("name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"Frame {image_id} has no non-empty 'name'.")
        dimensions = image_dimensions.get(file_name)
        if (
            not isinstance(dimensions, (tuple, list))
            or len(dimensions) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions)
        ):
            raise ValueError(f"Missing valid (width, height) for BDD100K frame '{file_name}'.")
        width, height = dimensions
        image_record = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "source_frame": {
                key: deepcopy(frame[key])
                for key in ("videoName", "frameIndex")
                if key in frame
            },
        }
        images.append(image_record)

        for label_index, label in enumerate(frame.get("labels", [])):
            box = label.get("box2d")
            if box is None:
                # BDD100K may include non-box labels (for example lane/drivable
                # area records); retain no fabricated detection for those.
                continue
            if not isinstance(box, dict):
                raise ValueError(f"Invalid box2d in frame '{file_name}', label {label_index}.")
            coords = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in coords
            ):
                raise ValueError(f"Invalid box coordinates in frame '{file_name}', label {label_index}.")
            x1, y1, x2, y2 = coords
            box_width, box_height = x2 - x1, y2 - y1
            if x1 < 0 or y1 < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Non-positive or negative-origin box in frame '{file_name}', label {label_index}.")
            if x2 > width or y2 > height:
                raise ValueError(f"Box exceeds supplied image dimensions for frame '{file_name}'.")
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_ids[label["category"]],
                "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "source_label": deepcopy(label),
            }
            annotations.append(annotation)

    return {
        "info": {"description": "BDD100K labels converted to NOORIX COCO-style metadata"},
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "noorix_source": {
            "dataset": source_dataset,
            "format": "bdd100k-frame-labels",
            "conversion": "metadata-only",
            "image_dimensions_source": "caller-supplied inventory",
        },
    }
