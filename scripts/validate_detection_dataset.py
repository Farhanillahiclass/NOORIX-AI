"""Validate a COCO-style object-detection annotation JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import validate_coco_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Path to COCO annotation JSON")
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image root; verifies each referenced image exists within it",
    )
    args = parser.parse_args(argv)

    errors = validate_coco_dataset(args.annotations, args.images_dir)
    if errors:
        print(f"Dataset validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Dataset structure validation passed.")
    if args.images_dir:
        print("Referenced image-file checks passed.")
    print("Note: this does not certify annotation correctness or model accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
