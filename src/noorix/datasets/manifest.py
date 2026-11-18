"""Validation helpers for the NOORIX dataset registry manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ALLOWED_STATUSES = {
    "candidate",
    "license_reviewed",
    "downloaded",
    "pending_validation",
    "validated",
    "ready_for_training",
    "rejected",
}


def validate_dataset_manifest(manifest_file: str | Path) -> list[str]:
    """Return structural/provenance errors for a dataset registry JSON file."""
    path = Path(manifest_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"Cannot read manifest file: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]

    if not isinstance(payload, dict):
        return ["Root JSON value must be an object."]

    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("'schema_version' must be 1.")
    if payload.get("format") != "coco_detection":
        errors.append("'format' must be 'coco_detection'.")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return errors + ["'datasets' must be an array."]

    seen_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        prefix = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string.")
        elif dataset_id in seen_ids:
            errors.append(f"{prefix}.id duplicates '{dataset_id}'.")
        else:
            seen_ids.add(dataset_id)

        for field in ("name", "release", "task", "source_url", "terms_url", "license_summary"):
            value = dataset.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string.")

        status = dataset.get("status")
        if status not in _ALLOWED_STATUSES:
            errors.append(f"{prefix}.status must be one of: {', '.join(sorted(_ALLOWED_STATUSES))}.")

        planned_splits = dataset.get("planned_splits")
        if not isinstance(planned_splits, list) or not planned_splits:
            errors.append(f"{prefix}.planned_splits must be a non-empty array.")
        elif any(not isinstance(split, str) or not split.strip() for split in planned_splits):
            errors.append(f"{prefix}.planned_splits entries must be non-empty strings.")

        restrictions = dataset.get("use_restrictions")
        if not isinstance(restrictions, list) or any(
            not isinstance(item, str) or not item.strip() for item in restrictions
        ):
            errors.append(f"{prefix}.use_restrictions must be an array of non-empty strings.")

    return errors


def validate_split_disjointness(splits: dict[str, list[str]]) -> list[str]:
    """Detect sample identifiers assigned to more than one named split."""
    errors: list[str] = []
    owners: dict[str, str] = {}
    for split_name, sample_ids in splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            errors.append("Split names must be non-empty strings.")
            continue
        if not isinstance(sample_ids, list):
            errors.append(f"Split '{split_name}' must contain an array of sample IDs.")
            continue
        local_seen: set[str] = set()
        for sample_id in sample_ids:
            if not isinstance(sample_id, str) or not sample_id.strip():
                errors.append(f"Split '{split_name}' contains an empty or non-string sample ID.")
                continue
            if sample_id in local_seen:
                errors.append(f"Sample '{sample_id}' is duplicated within split '{split_name}'.")
                continue
            local_seen.add(sample_id)
            previous = owners.get(sample_id)
            if previous is not None and previous != split_name:
                errors.append(
                    f"Sample '{sample_id}' leaks across splits '{previous}' and '{split_name}'."
                )
            else:
                owners[sample_id] = split_name
    return errors
