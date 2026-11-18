"""Deterministic quality and class-balance summaries for COCO annotations."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
from typing import Any


def _number_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def analyze_coco_quality(payload: dict[str, Any]) -> dict[str, Any]:
    """Return class distribution, annotation coverage and bbox statistics.

    This is a descriptive report, not an annotation-correctness or model-quality
    certification. Call validate_coco_dataset before treating the input as valid.
    """
    if not isinstance(payload, dict):
        raise ValueError("COCO dataset root must be an object.")
    for key in ("images", "categories", "annotations"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO dataset '{key}' must be an array.")

    images = payload["images"]
    categories = payload["categories"]
    annotations = payload["annotations"]

    category_names: dict[int, str] = {}
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or not isinstance(category.get("id"), int):
            raise ValueError(f"categories[{index}] must have an integer id.")
        if isinstance(category["id"], bool):
            raise ValueError(f"categories[{index}].id cannot be boolean.")
        if category["id"] in category_names:
            raise ValueError(f"Duplicate category id: {category['id']}.")
        category_names[category["id"]] = str(category.get("name", category["id"]))

    image_ids: set[int] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict) or not isinstance(image.get("id"), int) or isinstance(image.get("id"), bool):
            raise ValueError(f"images[{index}] must have an integer id.")
        if image["id"] in image_ids:
            raise ValueError(f"Duplicate image id: {image['id']}.")
        image_ids.add(image["id"])

    instance_counts: Counter[int] = Counter()
    per_class_images: dict[int, set[int]] = defaultdict(set)
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    seen_boxes: set[tuple[Any, ...]] = set()
    duplicate_box_count = 0
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotations[{index}] must be an object.")
        category_id = annotation.get("category_id")
        image_id = annotation.get("image_id")
        bbox = annotation.get("bbox")
        if not isinstance(category_id, int) or isinstance(category_id, bool) or category_id not in category_names:
            raise ValueError(f"annotations[{index}] references unknown category_id.")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id not in image_ids:
            raise ValueError(f"annotations[{index}] references unknown image_id.")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in bbox
            )
        ):
            raise ValueError(f"annotations[{index}].bbox must contain four finite numbers.")
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError(f"annotations[{index}].bbox dimensions must be positive.")
        instance_counts[category_id] += 1
        per_class_images[category_id].add(image_id)
        widths.append(float(width))
        heights.append(float(height))
        areas.append(float(width * height))
        box_key = (image_id, category_id, *bbox)
        if box_key in seen_boxes:
            duplicate_box_count += 1
        else:
            seen_boxes.add(box_key)

    class_distribution = []
    for category_id, name in sorted(category_names.items(), key=lambda item: (item[1].casefold(), item[0])):
        class_distribution.append({
            "category_id": category_id,
            "category_name": name,
            "instances": instance_counts[category_id],
            "images": len(per_class_images[category_id]),
        })

    nonzero_counts = [count for count in instance_counts.values() if count > 0]
    imbalance_ratio = (
        round(max(nonzero_counts) / min(nonzero_counts), 6)
        if nonzero_counts else None
    )
    images_with_annotations = len({annotation["image_id"] for annotation in annotations})
    image_count = len(images)
    return {
        "schema_version": 1,
        "summary": {
            "images": image_count,
            "categories": len(categories),
            "annotations": len(annotations),
            "images_with_annotations": images_with_annotations,
            "images_without_annotations": image_count - images_with_annotations,
            "annotation_coverage_percent": (
                round(images_with_annotations * 100 / image_count, 4) if image_count else None
            ),
            "classes_without_instances": sum(
                1 for category_id in category_names if instance_counts[category_id] == 0
            ),
            "class_imbalance_ratio_max_to_min": imbalance_ratio,
            "duplicate_box_annotations": duplicate_box_count,
        },
        "class_distribution": class_distribution,
        "bbox_statistics": {
            "width": _number_summary(widths),
            "height": _number_summary(heights),
            "area": _number_summary(areas),
        },
        "limitations": [
            "Descriptive statistics only; no claim of annotation correctness or model accuracy.",
            "Class imbalance ratio excludes categories with zero instances; zero-instance classes are reported separately.",
            "Duplicate boxes are exact matches by image, category and XYWH coordinates; near-duplicates are not detected.",
        ],
    }
