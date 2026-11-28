"""Repeatable, local CPU object-detection latency benchmark.

This benchmark uses one caller-supplied image, writes aggregate timing metadata,
and never saves or uploads the image or detection contents.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import stat
import sys
import time
from typing import Sequence


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = [float(value) for value in values]
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("values must all be finite numbers")
    ordered.sort()
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(samples_ms: Sequence[float]) -> dict[str, float | int]:
    if not samples_ms:
        raise ValueError("at least one latency sample is required")
    values = [float(value) for value in samples_ms]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("latency samples must be finite positive numbers")
    mean = statistics.fmean(values)
    return {
        "iterations": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "throughput_fps": 1000.0 / mean,
    }


def validate_positive_latency(value: float, label: str = "latency") -> float:
    """Validate one measured duration before including it in a report."""
    measured = float(value)
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return measured


def _is_symlink_or_junction(path: Path) -> bool:
    """Detect symbolic links and Windows directory junctions."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    # Windows reparse-point metadata catches junctions even on Python 3.11.
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _has_symlink_component(path: Path) -> bool:
    """Detect a symlink at the output path or any existing ancestor."""
    candidate = Path(path).absolute()
    return any(_is_symlink_or_junction(component) for component in (candidate, *candidate.parents))


def create_unique_report_directory(output_dir: Path, stamp: str) -> Path:
    """Create a collision-safe report directory without following symlinks."""
    if _has_symlink_component(output_dir):
        raise ValueError("benchmark report output path contains a symbolic link")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        name = stamp if suffix == 0 else f"{stamp}-{suffix}"
        candidate = output_dir / name
        if _has_symlink_component(candidate):
            raise ValueError("benchmark report directory contains a symbolic link")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def write_report_pair(json_path: Path, md_path: Path, payload: dict, markdown: str) -> None:
    """Write JSON and Markdown exclusively, cleaning only this call's partial output."""
    created_files: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8") as stream:
            created_files.append(json_path)
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        with md_path.open("x", encoding="utf-8") as stream:
            created_files.append(md_path)
            stream.write(markdown)
    except Exception:
        for created in created_files:
            try:
                created.unlink()
            except OSError:
                pass
        raise


MAX_CONFIG_BYTES = 1024 * 1024


def load_benchmark_config(path: Path) -> dict:
    """Read a bounded JSON config without following symlinked paths."""
    path = Path(path).expanduser()
    if _has_symlink_component(path):
        raise ValueError("configuration path contains a symbolic link")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("configuration path is not a regular file")
    if metadata.st_size > MAX_CONFIG_BYTES:
        raise ValueError("configuration exceeds the 1 MiB inspection limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("configuration exceeds the 1 MiB inspection limit")
    config = json.loads(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    return config

def safe_metadata_text(value: object, limit: int = 160) -> str:
    """Escape untrusted metadata for Markdown and keep it on one bounded line."""
    text = "".join(character if character.isprintable() else " " for character in str(value))
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in (chr(92), "|", "`", "*", "_", "[", "]", "!"):
        safe = safe.replace(character, chr(92) + character)
    return safe[:limit]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure local object-detection latency on a supplied image."
    )
    parser.add_argument("--config", default="config.json", help="Configuration JSON path (default: config.json)")
    parser.add_argument("--image", required=True, help="Local image path used for every inference")
    parser.add_argument("--iterations", type=int, default=30, help="Measured runs (1-1000; default: 30)")
    parser.add_argument("--warmup", type=int, default=5, help="Additional unmeasured warm-up runs (0-100; default: 5)")
    parser.add_argument("--output-dir", default="reports/benchmarks", help="Report directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.iterations <= 1000:
        print("--iterations must be between 1 and 1000", file=sys.stderr)
        return 2
    if not 0 <= args.warmup <= 100:
        print("--warmup must be between 0 and 100", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = root / image_path

    try:
        config = load_benchmark_config(config_path)
        detector_config = config.get("object_detection")
        if not isinstance(detector_config, dict):
            raise ValueError("config must contain an object_detection object")

        import cv2
        from noorix.objects import create_object_detector

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("image could not be decoded by OpenCV")

        detector = create_object_detector(detector_config, config_path=config_path)
        started = time.perf_counter()
        first_results = detector.detect(frame)
        cold_start_ms = (time.perf_counter() - started) * 1000.0
        cold_start_ms = validate_positive_latency(cold_start_ms, "cold-start timing")

        for _ in range(args.warmup):
            detector.detect(frame)

        samples = []
        final_results = first_results
        for _ in range(args.iterations):
            started = time.perf_counter()
            final_results = detector.detect(frame)
            samples.append((time.perf_counter() - started) * 1000.0)

        stats = summarize_latencies(samples)
        generated = datetime.now().astimezone().isoformat(timespec="seconds")
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        report_dir = create_unique_report_directory(output_dir, stamp)

        def package_version(distribution: str) -> str | None:
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                return None

        payload = {
            "project": "NOORIX-AI",
            "generated_at": generated,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": safe_metadata_text(platform.processor() or platform.uname().processor or "unknown"),
            "runtime_versions": {
                "opencv": getattr(cv2, "__version__", None),
                "numpy": package_version("numpy"),
                "onnxruntime": package_version("onnxruntime"),
            },
            "backend": safe_metadata_text(detector_config.get("backend", "opencv_dnn")),
            "confidence_threshold": detector_config.get("confidence_threshold"),
            "input_size": detector_config.get("input_size"),
            "model_filename": safe_metadata_text(Path(str(detector_config.get("model_path", ""))).name),
            "image_filename": safe_metadata_text(image_path.name),
            "image_dimensions": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "cold_start_ms": cold_start_ms,
            "warmup_runs": args.warmup,
            "latency": stats,
            "detections_last_run": len(final_results),
            "limitations": [
                "This is a repeatability aid, not a representative accuracy or safety evaluation.",
                "Repeated runs use one image and do not measure camera capture, speech, OCR, or end-to-end agent latency.",
                "Memory use and thermal throttling are not measured.",
                "Compare results only when hardware, model, image, config, power mode, and software versions are recorded consistently.",
            ],
        }
        json_path = report_dir / "benchmark-report.json"
        md_path = report_dir / "benchmark-report.md"
        lines = [
            "# NOORIX-AI CPU Object Detection Benchmark", "",
            f"- Generated: {generated}",
            f"- Platform: {payload['platform']}",
            f"- Processor: {payload['processor']}",
            f"- Python: {payload['python']}",
            f"- Backend: {payload['backend']}",
            f"- Runtime versions: {payload['runtime_versions']}",
            f"- Confidence threshold: {payload['confidence_threshold']}",
            f"- Input size: {payload['input_size']}",
            f"- Model file: {payload['model_filename']}",
            f"- Image file: {payload['image_filename']} ({frame.shape[1]}×{frame.shape[0]})",
            f"- Cold first inference: {cold_start_ms:.2f} ms",
            f"- Warm-up runs: {args.warmup}", "",
            "| Metric | Result |", "|---|---:|",
            f"| Measured iterations | {stats['iterations']} |",
            f"| Mean latency | {stats['mean_ms']:.2f} ms |",
            f"| Median latency | {stats['median_ms']:.2f} ms |",
            f"| P95 latency | {stats['p95_ms']:.2f} ms |",
            f"| Minimum latency | {stats['min_ms']:.2f} ms |",
            f"| Maximum latency | {stats['max_ms']:.2f} ms |",
            f"| Approx. throughput | {stats['throughput_fps']:.2f} FPS |",
            f"| Detections (last run) | {len(final_results)} |", "",
            "## Limitations", "",
        ]
        lines.extend(f"- {item}" for item in payload["limitations"])
        lines.append("")
        write_report_pair(json_path, md_path, payload, "\n".join(lines))
        print(f"Cold first inference: {cold_start_ms:.2f} ms")
        print(f"Mean: {stats['mean_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms | FPS: {stats['throughput_fps']:.2f}")
        print(f"JSON report: {safe_metadata_text(json_path.relative_to(root) if json_path.is_relative_to(root) else json_path)}")
        print(f"Markdown report: {safe_metadata_text(md_path.relative_to(root) if md_path.is_relative_to(root) else md_path)}")
        print("This single-image benchmark is not a production performance or safety certification.")
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Benchmark failed ({type(exc).__name__}). Check local config, model and image files.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
