"""Compare COCO annotation files across train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import analyze_coco_splits  # noqa: E402
from noorix.datasets import validate_coco_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="COCO JSON for the training split")
    parser.add_argument("--validation", type=Path, help="COCO JSON for the validation split")
    parser.add_argument("--test", type=Path, help="COCO JSON for the test split")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    paths = {
        name: path
        for name, path in (
            ("train", args.train),
            ("validation", args.validation),
            ("test", args.test),
        )
        if path is not None
    }
    if len(paths) < 2:
        parser.error("provide at least two split files: --train, --validation and/or --test")

    payloads = {}
    failed = False
    for split_name, path in paths.items():
        errors = validate_coco_dataset(path)
        if errors:
            failed = True
            print(f"Split '{split_name}' validation failed ({len(errors)} issue(s)):")
            for error in errors:
                print(f"- {error}")
            continue
        try:
            payloads[split_name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failed = True
            print(f"Split '{split_name}' could not be read: {exc}")

    if failed:
        print("Split analysis not generated because one or more inputs failed validation.")
        return 1

    report = analyze_coco_splits(payloads)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Split analysis report written to: {args.output}")
    else:
        print(rendered, end="")

    if report["cross_split_filename_overlap_count"]:
        print(
            "Warning: cross-split filename overlaps found: "
            f"{report['cross_split_filename_overlap_count']}"
        )
    else:
        print("No exact cross-split filename overlaps found.")
    print("Note: renamed/content-identical copies are not detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
