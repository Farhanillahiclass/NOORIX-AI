"""Generate a JSON quality report for a COCO-style detection dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import analyze_coco_quality  # noqa: E402
from noorix.datasets import validate_coco_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Path to COCO annotation JSON")
    parser.add_argument("--output", type=Path, help="Optional report JSON output path")
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image root; validate referenced image paths before analysis",
    )
    args = parser.parse_args(argv)

    errors = validate_coco_dataset(args.annotations, args.images_dir)
    if errors:
        print(f"Dataset validation failed ({len(errors)} issue(s)); quality report not generated:")
        for error in errors:
            print(f"- {error}")
        return 1

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    report = analyze_coco_quality(payload)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Quality report written to: {args.output}")
    else:
        print(rendered, end="")
    print("Note: descriptive statistics only; not an annotation or model-accuracy certification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
