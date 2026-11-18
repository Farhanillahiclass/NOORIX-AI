"""Convert supported detection annotation metadata to NOORIX COCO-style JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noorix.datasets import (  # noqa: E402
    adapt_bdd100k,
    adapt_coco,
    adapt_lvis,
    adapt_objects365,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("coco", "lvis", "objects365", "bdd100k"), required=True)
    parser.add_argument("input", type=Path, help="Source annotation JSON")
    parser.add_argument("output", type=Path, help="Destination COCO-style JSON")
    parser.add_argument(
        "--image-dimensions",
        type=Path,
        help="Required for BDD100K: JSON object mapping frame filename to [width, height]",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.format == "bdd100k":
        if args.image_dimensions is None:
            parser.error("--image-dimensions is required for BDD100K")
        dimensions = json.loads(args.image_dimensions.read_text(encoding="utf-8"))
        converted = adapt_bdd100k(payload, image_dimensions=dimensions)
    else:
        adapter = {
            "coco": adapt_coco,
            "lvis": adapt_lvis,
            "objects365": adapt_objects365,
        }[args.format]
        converted = adapter(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Converted annotation metadata written to: {args.output}")
    print("Next: run scripts/validate_detection_dataset.py on this output.")
    print("No image files were read, copied, or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
