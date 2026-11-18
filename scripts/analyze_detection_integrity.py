"""Generate a read-only image integrity and duplicate report for COCO splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import analyze_coco_integrity  # noqa: E402
from noorix.datasets import validate_coco_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}-annotations", type=Path)
        parser.add_argument(f"--{name}-images-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional report JSON output path")
    parser.add_argument("--verify-image-decode", action="store_true",
                         help="Verify image decoding with Pillow (install the camera extra)")
    args = parser.parse_args(argv)

    splits = {}
    failed = False
    for name in ("train", "validation", "test"):
        annotations = getattr(args, f"{name}_annotations")
        images_dir = getattr(args, f"{name}_images_dir")
        if annotations is None and images_dir is None:
            continue
        if annotations is None or images_dir is None:
            parser.error(f"--{name}-annotations and --{name}-images-dir must be supplied together")
        errors = validate_coco_dataset(annotations)
        if errors:
            failed = True
            print(f"Split '{name}' annotation validation failed ({len(errors)} issue(s)):")
            for error in errors:
                print(f"- {error}")
            continue
        try:
            payload = json.loads(annotations.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failed = True
            print(f"Split '{name}' annotation file could not be read: {exc}")
            continue
        splits[name] = {"payload": payload, "images_dir": images_dir}

    if not splits:
        parser.error("provide at least one complete split using annotation and image-root arguments")
    if failed:
        print("Integrity report not generated because annotation validation failed.")
        return 1
    try:
        report = analyze_coco_integrity(splits, verify_image_decode=args.verify_image_decode)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Integrity analysis could not run: {exc}")
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Integrity report written to: {args.output}")
    else:
        print(rendered, end="")
    print(
        "Summary: "
        f"{report['summary']['files_hashed']} hashed, "
        f"{report['summary']['duplicate_groups']} duplicate groups, "
        f"{report['summary']['missing_files']} missing, "
        f"{report['summary']['unreadable_files']} unreadable, "
        f"{report['summary']['decode_failures']} decode failures."
    )
    print("Note: exact byte-level duplicates only; files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
