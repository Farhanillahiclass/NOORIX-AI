"""Validate a NOORIX dataset registry manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import validate_dataset_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "datasets/object_detection/splits/manifest.json",
        help="Path to the dataset registry JSON",
    )
    args = parser.parse_args(argv)

    errors = validate_dataset_manifest(args.manifest)
    if errors:
        print(f"Manifest validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Dataset manifest structure validation passed.")
    print("Note: this does not grant dataset rights or certify data quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
