"""Split-level class distribution and filename leakage analysis."""

from __future__ import annotations

from typing import Any

from .quality import analyze_coco_quality


def analyze_coco_splits(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize COCO splits and report exact cross-split filename overlap.

    Filenames are normalized for slash direction and case. Numeric COCO image IDs
    are deliberately not compared across files because exporters commonly restart
    IDs independently for each split.
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
    for split_name, report in split_reports.items():
        counts = {
            item["category_name"]: item["instances"]
            for item in report["class_distribution"]
        }
        distribution[split_name] = {
            name: counts.get(name, 0) for name in sorted(class_names)
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
            }
            for name, report in split_reports.items()
        },
        "class_names": sorted(class_names),
        "cross_split_filename_overlaps": overlaps,
        "cross_split_filename_overlap_count": len(overlaps),
        "limitations": [
            "Filename overlap is exact after slash normalization and case-folding; renamed duplicate images are not detected.",
            "Numeric image IDs are not compared across split files because IDs may be independently assigned.",
            "Class counts describe annotations and do not establish label correctness or model performance.",
        ],
    }
