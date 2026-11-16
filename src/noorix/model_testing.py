"""Sequential local smoke tests for every configured NOORIX model.

These checks validate loadability and basic inference only. They do not measure
real-world accuracy or certify a production deployment.
"""
from __future__ import annotations

import hashlib
import gc
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

from noorix.paths import resolve_config_path


@dataclass
class TestResult:
    name: str
    status: str
    duration_seconds: float
    message: str
    details: dict[str, Any] | None = None


def _run_one(name: str, operation: Callable[[], tuple[str, dict | None]]) -> TestResult:
    started = time.perf_counter()
    try:
        status, details = operation()
        message = {
            "PASS": "Smoke test completed.",
            "SKIPPED": "Required local file or dependency is unavailable.",
            "WARN": "Test completed with a limitation.",
        }.get(status, "Test completed.")
        return TestResult(name, status, round(time.perf_counter() - started, 3), message, details)
    except Exception as exc:
        return TestResult(name, "FAIL", round(time.perf_counter() - started, 3),
                          f"{type(exc).__name__}: {exc}")


def _load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}. Copy config.example.json to config.json.")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config root must be a JSON object.")
    return value


def _test_package(config: dict, config_path: Path):
    model_root = Path(os.environ.get("NOORIX_MODELS_DIR", config_path.parent)).expanduser()
    manifest_path = model_root / "manifest.json"
    checksum_path = model_root / "checksums.sha256"
    if not manifest_path.is_file() and not checksum_path.is_file():
        return "SKIPPED", {"reason": "No manifest.json or checksums.sha256 beside config."}
    checked, failures = 0, []
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                failures.append(f"Malformed checksum line: {line[:80]}")
                continue
            expected, relative = parts[0], parts[1].lstrip("* ")
            target = config_path.parent / relative
            if not target.is_file():
                failures.append(f"Missing: {relative}")
                continue
            hasher = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            checked += 1
            if digest.lower() != expected.lower():
                failures.append(f"Checksum mismatch: {relative}")
    if failures:
        raise RuntimeError("; ".join(failures[:10]))
    return "PASS", {"manifest": str(manifest_path) if manifest_path.is_file() else None,
                    "checksum_files_verified": checked}


def _test_currency(config: dict, config_path: Path):
    settings = config.get("currency", {})
    if not isinstance(settings, dict) or not settings.get("model_path"):
        return "SKIPPED", {"reason": "currency.model_path is not configured"}
    model = resolve_config_path(settings["model_path"], config_path)
    if not model.is_file():
        return "SKIPPED", {"reason": f"Model missing: {model}"}
    backend = settings.get("backend", "onnx")
    import numpy as np
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    if backend == "pytorch_mobilenet_v2":
        from noorix.pkr_currency import PytorchPKRCurrencyEstimator
        labels = settings.get("labels_path")
        if not labels:
            raise ValueError("currency.labels_path is required for PyTorch backend")
        adapter = PytorchPKRCurrencyEstimator(
            model, resolve_config_path(labels, config_path),
            input_size=int(settings.get("input_size", 224)),
        )
    elif backend == "onnx":
        from noorix.currency import CurrencyEstimator
        labels = settings.get("labels")
        if not labels:
            raise ValueError("ONNX currency config requires labels")
        adapter = CurrencyEstimator(model, labels, input_size=int(settings.get("input_size", 224)))
    else:
        raise ValueError(f"Unsupported currency backend: {backend}")
    prediction = adapter.predict(image)
    confidence = float(prediction["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise RuntimeError("Currency confidence is outside [0, 1]")
    return "PASS", {"backend": backend, "confidence": round(confidence, 4),
                    "note": "Blank synthetic image tests inference only, not note accuracy."}


def _test_objects(config: dict, config_path: Path):
    settings = config.get("object_detection", {})
    if not isinstance(settings, dict) or not settings.get("model_path"):
        return "SKIPPED", {"reason": "object_detection.model_path is not configured"}
    model = resolve_config_path(settings["model_path"], config_path)
    if not model.is_file():
        return "SKIPPED", {"reason": f"Model missing: {model}"}
    from noorix.objects import create_object_detector
    import numpy as np
    detector = create_object_detector(settings, config_path)
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    if not isinstance(detections, list):
        raise RuntimeError("Detector did not return a list.")
    return "PASS", {"backend": settings.get("backend", "opencv_dnn"),
                    "detections_on_blank_test_frame": len(detections),
                    "note": "Blank synthetic image tests inference only, not detection accuracy."}


def _test_ocr(config: dict):
    settings = config.get("ocr", {})
    language = settings.get("language", "eng") if isinstance(settings, dict) else "eng"
    if not shutil.which("tesseract"):
        return "SKIPPED", {"reason": "Tesseract executable not found on PATH."}
    try:
        import pytesseract
        from PIL import Image, ImageDraw
    except ImportError as exc:
        return "SKIPPED", {"reason": f"OCR Python dependency unavailable: {exc}"}
    installed = set(pytesseract.get_languages(config=""))
    missing = sorted(set(language.split("+")) - installed)
    if missing:
        return "SKIPPED", {"reason": "Tesseract language data missing: " + ", ".join(missing)}
    if language == "urd":
        return "WARN", {"language": language, "note": "Urdu trained-data is installed; automated sample OCR is not run without a bundled verified Urdu sample font."}
    image = Image.new("RGB", (720, 180), "white")
    ImageDraw.Draw(image).text((25, 55), "NOORIX TEST 123", fill="black")
    from noorix.ocr import PrintedEnglishOCR
    text = PrintedEnglishOCR(language=language).read(image)
    if not text.strip():
        raise RuntimeError("OCR returned empty text for a generated test image.")
    return "PASS", {"language": language, "sample_text": text[:120]}


def _test_asr(config: dict, config_path: Path):
    settings = config.get("asr", {})
    if not isinstance(settings, dict) or not settings.get("model_path"):
        return "SKIPPED", {"reason": "asr.model_path is not configured"}
    model = resolve_config_path(settings["model_path"], config_path)
    processor_value = settings.get("processor_path")
    processor = resolve_config_path(processor_value, config_path) if processor_value else model
    if not model.is_dir() or not processor.is_dir():
        return "SKIPPED", {"reason": "Whisper model or processor directory is missing"}
    import numpy as np
    from noorix.whisper_asr import WhisperASR
    adapter = WhisperASR(
        model, language=str(settings.get("language", "urdu")),
        max_new_tokens=min(int(settings.get("max_new_tokens", 32)), 32),
        processor_path=processor,
    )
    transcript = adapter.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    return "PASS", {"language": settings.get("language", "urdu"),
                    "silence_transcription": transcript[:120],
                    "note": "Silence smoke test only; evaluate real labelled recordings separately."}


def _test_tts():
    try:
        from noorix.speech import SpeechOutput
        voices = SpeechOutput.available_voices()
    except RuntimeError as exc:
        return "SKIPPED", {"reason": str(exc)}
    if not voices:
        return "SKIPPED", {"reason": "No local system speech voices are installed."}
    return "PASS", {"available_voice_count": len(voices),
                    "voices": [v.get("name", "unknown") for v in voices[:8]],
                    "note": "Voice enumeration only; no audio is played."}


def run_all_tests(config_path="config.json", report_dir="reports/model-tests", progress=None):
    """Run package, PKR, YOLO, OCR, Whisper and TTS smoke tests sequentially."""
    config_path = Path(config_path).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    results = []

    def add(name, operation):
        result = _run_one(name, operation)
        results.append(result)
        gc.collect()
        if progress:
            progress(result)

    try:
        config = _load_config(config_path)
        add("Model package integrity", lambda: _test_package(config, config_path))
        add("PKR currency model", lambda: _test_currency(config, config_path))
        add("YOLO object detection", lambda: _test_objects(config, config_path))
        add("Printed OCR", lambda: _test_ocr(config))
        add("Whisper speech-to-text", lambda: _test_asr(config, config_path))
        add("Local text-to-speech", _test_tts)
    except Exception as exc:
        results.append(TestResult("Configuration", "FAIL", 0.0, f"{type(exc).__name__}: {exc}"))

    summary = {
        "project": "NOORIX-AI",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "results": [asdict(item) for item in results],
        "totals": {status.lower(): sum(item.status == status for item in results)
                   for status in ("PASS", "FAIL", "SKIPPED", "WARN")},
        "production_readiness": "NOT_CERTIFIED",
        "disclaimer": "Smoke tests validate local loading/basic execution only. They do not certify model accuracy, accessibility safety, or production readiness.",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"noorix_model_test_{stamp}.json"
    md_path = report_dir / f"noorix_model_test_{stamp}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# NOORIX-AI Model Test Report", "", f"- Started (UTC): {started_at}", "",
             "| Test | Status | Seconds | Result |", "|---|---:|---:|---|"]
    for item in results:
        lines.append(f"| {item.name} | {item.status} | {item.duration_seconds:.3f} | {item.message.replace('|', '/')} |")
        if item.details:
            lines.extend(["", f"**{item.name} details:** {json.dumps(item.details, ensure_ascii=False)}"])
    lines.extend(["", f"**Totals:** {json.dumps(summary['totals'])}", "",
                  "**Production status:** NOT CERTIFIED — real labelled data, device, camera, usability and safety validation remain required."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["report_json"] = str(json_path)
    summary["report_markdown"] = str(md_path)
    return summary
