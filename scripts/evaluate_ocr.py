#!/usr/bin/env python
"""Evaluate printed OCR against a human-verified CSV manifest.

CSV columns: image,reference,language. Image paths are relative to the manifest.
Use only consented, lawfully sourced images.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from noorix.evaluation import character_error_counts, error_rate, word_error_counts


def load_manifest(path: Path) -> list[dict[str, str]]:
    required = {"image", "reference", "language"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain image,reference,language columns")
        rows = []
        for line, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            language = (row.get("language") or "").strip()
            if not image_value or language not in {"eng", "urd", "eng+urd"}:
                raise ValueError(f"Invalid image or language on CSV line {line}")
            rows.append({"image": image_value, "reference": row.get("reference") or "", "language": language})
    if not rows:
        raise ValueError("Manifest contains no evaluation samples")
    return rows


def evaluate(manifest: Path) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install OCR dependencies with: pip install -e '.[ocr]'") from exc
    from noorix.ocr import PrintedEnglishOCR

    rows = load_manifest(manifest)
    adapters = {}
    totals = defaultdict(lambda: {"char_errors": 0, "chars": 0, "word_errors": 0, "words": 0, "samples": 0})
    samples = []
    for row in rows:
        image_path = (manifest.parent / row["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image from manifest not found: {image_path}")
        language = row["language"]
        if language not in adapters:
            adapters[language] = PrintedEnglishOCR(language=language)
        with Image.open(image_path) as image:
            hypothesis = adapters[language].read(image)
        reference = row["reference"]
        char_errors, chars = character_error_counts(reference, hypothesis)
        word_errors, words = word_error_counts(reference, hypothesis)
        bucket = totals[language]
        bucket["char_errors"] += char_errors
        bucket["chars"] += chars
        bucket["word_errors"] += word_errors
        bucket["words"] += words
        bucket["samples"] += 1
        samples.append({"image": row["image"], "language": language, "reference": reference,
                        "hypothesis": hypothesis, "character_errors": char_errors,
                        "reference_characters": chars, "word_errors": word_errors,
                        "reference_words": words})

    by_language = {}
    for language, values in sorted(totals.items()):
        by_language[language] = {**values, "cer": error_rate(values["char_errors"], values["chars"]),
                                 "wer": error_rate(values["word_errors"], values["words"])}
    all_values = {key: sum(values[key] for values in totals.values())
                  for key in ("char_errors", "chars", "word_errors", "words", "samples")}
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "NOORIX PrintedEnglishOCR", "python": sys.version.split()[0],
            "platform": platform.platform(), "manifest": manifest.name,
            "sample_count": len(samples),
            "aggregate": {**all_values, "cer": error_rate(all_values["char_errors"], all_values["chars"]),
                          "wer": error_rate(all_values["word_errors"], all_values["words"])},
            "by_language": by_language, "samples": samples,
            "limitations": ["Metrics are only as representative as the supplied, human-verified manifest.",
                            "CER removes whitespace and preserves case and punctuation.",
                            "WER uses whitespace-delimited tokens and preserves case and punctuation.",
                            "This report is not a safety certification or a guarantee of field performance."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with image,reference,language columns")
    parser.add_argument("--output", type=Path, default=Path("reports/ocr-evaluation.json"))
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(f"Evaluated {report['sample_count']} samples")
    print(f"CER: {report['aggregate']['cer']}")
    print(f"WER: {report['aggregate']['wer']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
