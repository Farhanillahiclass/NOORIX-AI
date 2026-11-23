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


def _has_symlink_component(path: Path) -> bool:
    """Detect a symlink at the output path or any existing ancestor."""
    candidate = Path(path).absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))




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
        print(f"JSON report: {json_path.relative_to(root) if json_path.is_relative_to(root) else json_path}")
        print(f"Markdown report: {md_path.relative_to(root) if md_path.is_relative_to(root) else md_path}")
        print("This single-image benchmark is not a production performance or safety certification.")
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Benchmark failed ({type(exc).__name__}). Check local config, model and image files.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
