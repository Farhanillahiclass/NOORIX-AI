"""One-command local validation; no GitHub or network calls are made."""
from __future__ import annotations
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "node_modules", "build", "dist", "reports", ".noorix", ".idea", ".vscode"}
MAX_OUTPUT_CHARS = 12000
TIMEOUT_SECONDS = 1800


def validate_notebook_document(document: Any) -> str | None:
    """Return a structural error, or None for valid nbformat-4 JSON."""
    if not isinstance(document, dict):
        return "top-level JSON value must be an object"
    if document.get("nbformat") != 4:
        return "nbformat must be 4"
    if not isinstance(document.get("nbformat_minor"), int):
        return "nbformat_minor must be an integer"
    cells = document.get("cells")
    if not isinstance(cells, list):
        return "cells must be a list"
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            return f"cell {index} must be an object"
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            return f"cell {index} has an unsupported cell_type"
        source = cell.get("source")
        if not isinstance(source, (str, list)):
            return f"cell {index} source must be a string or list"
        if isinstance(source, list) and not all(isinstance(line, str) for line in source):
            return f"cell {index} source list must contain only strings"
        if cell.get("cell_type") == "code":
            if not isinstance(cell.get("outputs", []), list):
                return f"code cell {index} outputs must be a list"
            count = cell.get("execution_count")
            if count is not None and not isinstance(count, int):
                return f"code cell {index} execution_count must be an integer or null"
    return None


def _result(name: str, status: str, summary: str, output: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "summary": summary, "output": output[-MAX_OUTPUT_CHARS:]}


def _run(name: str, command: list[str], root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(command, cwd=root, env=env, text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return _result(name, "FAIL", f"timed out after {TIMEOUT_SECONDS}s", output)
    except OSError as exc:
        return _result(name, "FAIL", f"could not start command: {exc}")
    return _result(name, "PASS" if proc.returncode == 0 else "FAIL", f"exit code {proc.returncode}", proc.stdout or "")


def _notebook_check(root: Path) -> dict[str, str]:
    excluded = EXCLUDED_DIRS
    paths = []
    for path in root.rglob("*.ipynb"):
        relative = path.relative_to(root)
        if path.is_file() and not any(part in excluded for part in relative.parts) and not any(part.startswith(".") and part != ".github" for part in relative.parts):
            paths.append(path)
    if not paths:
        return _result("Notebook JSON", "WARN", "no notebooks found")
    failures = []
    for path in sorted(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            error = validate_notebook_document(data)
            if error:
                failures.append(f"{path.relative_to(root)}: {error}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"{path.relative_to(root)}: {exc}")
    if failures:
        return _result("Notebook JSON", "FAIL", f"{len(failures)} invalid of {len(paths)}", "\n".join(failures))
    return _result("Notebook JSON", "PASS", f"{len(paths)} valid nbformat-4 notebook(s)")


def validate_example_config(document: Any) -> str | None:
    """Validate supported config sections and safety-critical numeric bounds."""
    if not isinstance(document, dict):
        return "configuration root must be an object"

    detection = document.get("object_detection")
    if detection is not None:
        if not isinstance(detection, dict):
            return "object_detection must be an object"
        backend = detection.get("backend")
        if backend is not None and (not isinstance(backend, str) or not backend.strip()):
            return "object_detection.backend must be a non-empty string"
        for key in ("model_path", "labels_path"):
            value = detection.get(key)
            if value is not None and not isinstance(value, str):
                return f"object_detection.{key} must be a string"
        for key in ("confidence_threshold", "iou_threshold"):
            value = detection.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    return f"object_detection.{key} must be a number between 0 and 1"
        for key in ("input_size", "max_detections"):
            value = detection.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                return f"object_detection.{key} must be a positive integer"

    ocr = document.get("ocr")
    if ocr is not None:
        if not isinstance(ocr, dict):
            return "ocr must be an object"
        language = ocr.get("language")
        if language is not None and (not isinstance(language, str) or language not in {"eng", "urd", "eng+urd"}):
            return "ocr.language must be eng, urd, or eng+urd"

    currency = document.get("currency")
    if currency is not None:
        if not isinstance(currency, dict):
            return "currency must be an object"
        labels = currency.get("labels")
        if labels is not None and (not isinstance(labels, list) or not all(isinstance(label, str) for label in labels)):
            return "currency.labels must be a list of strings"
        input_size = currency.get("input_size")
        if input_size is not None and (isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0):
            return "currency.input_size must be a positive integer"
    return None


def validate_label_map(document: Any) -> str | None:
    """Validate the structural contract for a COCO-80 label map."""
    if not isinstance(document, list):
        return "label map root must be a JSON list"
    if len(document) != 80:
        return f"YOLOv8 COCO label map must contain exactly 80 labels, got {len(document)}"
    if any(not isinstance(label, str) or not label.strip() for label in document):
        return "label map entries must be non-empty strings"
    if len(set(document)) != 80:
        return "label map entries must be unique"
    return None


def _config_check(root: Path) -> dict[str, str]:
    files = sorted(root.glob("config.example*.json"))
    if not files:
        return _result("Example config JSON", "WARN", "no root config.example*.json found")
    errors = []
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            error = validate_example_config(document)
            if error:
                errors.append(f"{path.name}: {error}")
                continue
            detection = document.get("object_detection", {})
            labels_path = detection.get("labels_path") if isinstance(detection, dict) else None
            if labels_path:
                label_file = (root / labels_path).resolve()
                if not label_file.is_relative_to(root.resolve()):
                    errors.append(f"{path.name}: object_detection.labels_path escapes the project root")
                    continue
                try:
                    label_document = json.loads(label_file.read_text(encoding="utf-8"))
                    label_error = validate_label_map(label_document) if detection.get("backend") == "yolov8_onnx" else None
                    if label_error:
                        errors.append(f"{path.name}: {label_error}")
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append(f"{path.name}: object_detection.labels_path could not be read: {exc}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        return _result("Example config JSON", "FAIL", f"{len(errors)} invalid file(s)", "\n".join(errors))
    return _result("Example config JSON", "PASS", f"{len(files)} example config file(s) and configured YOLO label maps validated")


def _model_integrity_check(root: Path) -> list[dict[str, str]]:
    imported = root / "models" / "imported"
    manifests = sorted(imported.rglob("import_manifest.json")) if imported.is_dir() else []
    if not manifests:
        return [_result("Imported model integrity", "WARN", "no imported model manifests found")]
    results = []
    verifier = root / "scripts" / "verify_model_integrity.py"
    for manifest in manifests:
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            integrity = document.get("integrity", {}) if isinstance(document, dict) else {}
            if not isinstance(integrity, dict) or not integrity.get("files"):
                results.append(_result("Imported model integrity", "WARN",
                                       f"{manifest.parent.relative_to(root)} has no SHA-256 inventory"))
                continue
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            results.append(_result("Imported model integrity", "FAIL",
                                   f"{manifest.parent.relative_to(root)} manifest unreadable: {exc}"))
            continue
        results.append(_run(
            f"Model SHA-256: {manifest.parent.relative_to(root)}",
            [sys.executable, str(verifier), "--root", str(manifest.parent), "--manifest", str(manifest)],
            root,
        ))
    return results


def _offline_bundle_check(root: Path) -> dict[str, str]:
    """Resolve an optional local wheelhouse without network access or installation."""
    wheelhouse = root / "wheelhouse"
    if not wheelhouse.is_dir():
        return _result("Offline wheelhouse", "INFO", "no local wheelhouse supplied; optional check not run")
    return _run(
        "Offline wheelhouse",
        [sys.executable, str(root / "scripts" / "verify_offline_bundle.py"),
         "--wheelhouse", str(wheelhouse), "--extras", "all,dev"],
        root,
    )


def _tesseract_check() -> dict[str, str]:
    exe = shutil.which("tesseract")
    if not exe:
        return _result("Optional Tesseract", "WARN", "not installed; OCR runtime tests may skip")
    try:
        proc = subprocess.run([exe, "--list-langs"], text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result("Optional Tesseract", "WARN", str(exc))
    if proc.returncode:
        return _result("Optional Tesseract", "WARN", "language discovery failed", proc.stdout or "")
    langs = [line.strip() for line in (proc.stdout or "").splitlines()
             if line.strip() and not line.lower().startswith("list of available")]
    return _result("Optional Tesseract", "INFO", "languages: " + (", ".join(langs) if langs else "none reported"))


def _write_reports(root: Path, results: list[dict[str, str]]) -> tuple[Path, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    base = root / "reports" / "validation" / stamp
    target, suffix = base, 1
    while target.exists():
        target = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    target.mkdir(parents=True)
    counts = {s.lower(): sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "WARN", "INFO")}
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {"project": "NOORIX-AI", "generated_at": generated, "python": sys.version,
               "platform": platform.platform(), "results": results, "summary": counts}
    json_path, md_path = target / "validation-report.json", target / "validation-report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# NOORIX-AI Local Validation", "", f"- Generated: {generated}",
             f"- Python: {sys.version.split()[0]}", f"- Platform: {platform.platform()}", "",
             "| Check | Status | Result |", "|---|---|---|"]
    for item in results:
        summary = item["summary"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['name']} | **{item['status']}** | {summary} |")
    lines += ["", "## Summary", "", f"- PASS: {counts['pass']}", f"- FAIL: {counts['fail']}",
              f"- WARN: {counts['warn']}", f"- INFO: {counts['info']}", "",
              "Automated PASS results do not certify recognition accuracy, safe navigation, privacy compliance, legal rights, or production readiness.", ""]
    for item in results:
        if item["output"].strip():
            lines += [f"## {item['name']} output", "", "```text", item["output"].rstrip(), "```", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = [_result("Python version", "PASS" if sys.version_info >= (3, 11) else "FAIL",
                       f"{sys.version.split()[0]} (Python 3.11+ required)")]
    targets = [name for name in ("src", "tests", "scripts") if (root / name).exists()]
    if targets:
        results.append(_run("Compile source, tests and scripts", [sys.executable, "-m", "compileall", "-q", *targets], root))
    else:
        results.append(_result("Compile source and tests", "WARN", "no src/ or tests/ directory found"))
    if importlib.util.find_spec("pytest") is None:
        results.append(_result("Full pytest suite", "FAIL", 'pytest missing; install: python -m pip install -e ".[all,dev]"'))
    else:
        results.append(_run("Full pytest suite", [sys.executable, "-m", "pytest", "-q"], root))
    results.append(_run("Dependency consistency", [sys.executable, "-m", "pip", "check"], root))
    results.append(_run(\n        "Declared dependency audit",\n        [sys.executable, str(root / "scripts" / "dependency_audit.py")],\n        root,\n    ))
    results.append(_run(
        "Installed package capability check",
        [sys.executable, "-m", "noorix", "--check"],
        root,
    ))
    results.extend([_config_check(root), _notebook_check(root), *_model_integrity_check(root), _offline_bundle_check(root), _tesseract_check()])
    try:
        json_path, md_path = _write_reports(root, results)
    except OSError as exc:
        print(f"Could not write reports: {exc}", file=sys.stderr)
        return 2
    print("\nNOORIX-AI LOCAL VALIDATION")
    print("=" * 68)
    for item in results:
        print(f"[{item['status']:<4}] {item['name']}: {item['summary']}")
    failures = sum(item["status"] == "FAIL" for item in results)
    warnings = sum(item["status"] == "WARN" for item in results)
    print("-" * 68)
    print(f"FAIL: {failures} | WARN: {warnings}")
    print(f"JSON report: {json_path.relative_to(root)}")
    print(f"Markdown report: {md_path.relative_to(root)}")
    print("Automated PASS results are not production-readiness certification.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
