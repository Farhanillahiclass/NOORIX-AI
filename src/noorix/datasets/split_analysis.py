"""Split-level class distribution and filename leakage analysis."""

from __future__ import annotations

from typing import Any

from .quality import analyze_coco_quality


def analyze_coco_splits(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize COCO splits, class proportions, and filename overlap.

    Class percentages are proportions of annotations within each split, not
    proportions of images. Empty splits report None percentages. Filename
    matching normalizes slash direction and case. Numeric COCO image IDs are
    deliberately not compared because exporters commonly restart IDs per split.
    """
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a non-empty mapping of split names to COCO objects.")
    if any(not isinstance(name, str) or not name.strip() for name in splits):
        raise ValueError("split names must be non-empty strings.")

    split_reports: dict[str, Any] = {}
    filename_owners: dict[str, set[str]] = {}
    class_names: set[str] = set()

    for split_name in sorted(splits):
        payload = splits[split_name]
        report = analyze_coco_quality(payload)
        split_reports[split_name] = report
        for item in report["class_distribution"]:
            class_names.add(item["category_name"])
        for image in payload["images"]:
            filename = image.get("file_name")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(
                    f"Split '{split_name}' contains an image without a non-empty file_name."
                )
            normalized = filename.replace("\\", "/").strip().casefold()
            filename_owners.setdefault(normalized, set()).add(split_name)

    overlaps = [
        {"file_name": filename, "splits": sorted(owners)}
        for filename, owners in sorted(filename_owners.items())
        if len(owners) > 1
    ]

    distribution: dict[str, dict[str, int]] = {}
    percentages: dict[str, dict[str, float | None]] = {}
    for split_name, report in split_reports.items():
        counts = {
            item["category_name"]: item["instances"]
            for item in report["class_distribution"]
        }
        distribution[split_name] = {
            name: counts.get(name, 0) for name in sorted(class_names)
        }
        annotation_total = report["summary"]["annotations"]
        percentages[split_name] = {
            name: (
                round(distribution[split_name][name] / annotation_total * 100, 4)
                if annotation_total
                else None
            )
            for name in sorted(class_names)
        }

    comparison: dict[str, dict[str, Any]] = {}
    for class_name in sorted(class_names):
        class_percentages = {
            split_name: percentages[split_name][class_name]
            for split_name in sorted(percentages)
        }
        observed = [value for value in class_percentages.values() if value is not None]
        comparison[class_name] = {
            "instances_by_split": {
                split_name: distribution[split_name][class_name]
                for split_name in sorted(distribution)
            },
            "percentage_by_split": class_percentages,
            "spread_percentage_points": (
                round(max(observed) - min(observed), 4) if len(observed) >= 2 else None
            ),
        }

    return {
        "schema_version": 1,
        "splits": {
            name: {
                "images": report["summary"]["images"],
                "annotations": report["summary"]["annotations"],
                "images_with_annotations": report["summary"]["images_with_annotations"],
                "images_without_annotations": report["summary"]["images_without_annotations"],
                "class_distribution": distribution[name],
                "class_distribution_percentages": percentages[name],
            }
            for name, report in split_reports.items()
        },
        "class_names": sorted(class_names),
        "class_distribution_comparison": comparison,
        "cross_split_filename_overlaps": overlaps,
        "cross_split_filename_overlap_count": len(overlaps),
        "limitations": [
            "Class percentages are annotation-level proportions within each split, not image-level prevalence.",
            "A percentage-point spread is descriptive and does not establish that a split is statistically representative.",
            "Filename overlap is exact after slash normalization and case-folding; renamed duplicate images are not detected.",
            "Numeric image IDs are not compared across split files because IDs may be independently assigned.",
            "Class counts describe annotations and do not establish label correctness or model performance.",
        ],
    }
