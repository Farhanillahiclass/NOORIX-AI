"""One-command local validation; no GitHub or network calls are made."""
from __future__ import annotations
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "node_modules", "build", "dist", "reports", ".noorix", ".idea", ".vscode"}
MAX_OUTPUT_CHARS = 12000
MAX_NOTEBOOK_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 1800
_ANSI_ESCAPE = re.compile(r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-_])")


def _clean_diagnostic_text(value: Any, *, single_line: bool = False) -> str:
    """Remove terminal control sequences from untrusted tool and file metadata."""
    text = _ANSI_ESCAPE.sub("", str(value))
    if single_line:
        printable = "".join(character if character.isprintable() else " " for character in text)
        return " ".join(printable.split())[:300]
    return "".join(
        character if character in "\n\t" or character.isprintable() else " "
        for character in text
    )


def _safe_path_label(path: str | Path, root: Path) -> str:
    """Return a bounded project-relative label without terminal controls."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        absolute = Path(os.path.abspath(candidate))
        relative = absolute.relative_to(Path(os.path.abspath(root))).as_posix()
    except (OSError, RuntimeError, ValueError):
        return "<unavailable path>"
    return _clean_diagnostic_text(relative, single_line=True)[:240]


def python_version_supported(version: tuple[int, ...]) -> bool:
    """The currently CI-verified interpreter range is Python 3.11 through 3.13."""
    return (3, 11) <= version[:2] < (3, 14)


def validate_notebook_document(document: Any) -> str | None:
    """Return a structural error, or None for valid nbformat-4 JSON."""
    if not isinstance(document, dict):
        return "top-level JSON value must be an object"
    if document.get("nbformat") != 4:
        return "nbformat must be 4"
    minor = document.get("nbformat_minor")
    if isinstance(minor, bool) or not isinstance(minor, int) or minor < 0:
        return "nbformat_minor must be a non-negative integer"
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
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                return f"code cell {index} execution_count must be a non-negative integer or null"
    return None


def _result(name: str, status: str, summary: str, output: str = "") -> dict[str, str]:
    return {
        "name": _clean_diagnostic_text(name, single_line=True)[:160],
        "status": status,
        "summary": _clean_diagnostic_text(summary, single_line=True),
        "output": _clean_diagnostic_text(output)[-MAX_OUTPUT_CHARS:],
    }


def _redact_local_paths(text: str, root: Path) -> str:
    """Redact project and home directory prefixes before writing local reports."""
    candidates: list[tuple[str, str]] = []
    for path, replacement in ((root, "<PROJECT>"), (Path.home(), "<HOME>")):
        try:
            value = str(path)
            if value:
                candidates.extend([
                    (value, replacement),
                    (path.as_posix(), replacement),
                    (value.replace("/", chr(92)), replacement),
                ])
        except (OSError, RuntimeError):
            continue
    # Longest first prevents a shorter home prefix consuming a project prefix.
    for value, replacement in sorted(set(candidates), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(value), replacement, text, flags=re.IGNORECASE)
    return text


def _has_symlink_component(path: Path, root: Path) -> bool:
    """Return true for symlinks in a project-relative path; fail closed."""
    try:
        relative = path.absolute().relative_to(root.absolute())
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return True
        return False
    except (OSError, RuntimeError, ValueError):
        return True


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
        return _result(name, "FAIL", f"timed out after {TIMEOUT_SECONDS}s", _redact_local_paths(output, root))
    except OSError as exc:
        return _result(name, "FAIL", f"could not start command ({type(exc).__name__})")
    return _result(name, "PASS" if proc.returncode == 0 else "FAIL", f"exit code {proc.returncode}", _redact_local_paths(proc.stdout or "", root))



def discover_notebooks(root: Path) -> tuple[list[Path], list[str]]:
    """Find notebooks without following symlinked directories; report scan errors."""
    paths: list[Path] = []
    errors: list[str] = []

    def onerror(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        try:
            relative = _safe_path_label(filename, root) if filename else "<unknown>"
        except (OSError, ValueError, TypeError):
            relative = "<unavailable path>"
        errors.append(f"{relative}: directory scan failed ({type(error).__name__})")

    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
            current_path = Path(current)
            directories[:] = [
                name for name in directories
                if name not in EXCLUDED_DIRS
                and not (name.startswith(".") and name != ".github")
                and not (current_path / name).is_symlink()
            ]
            for name in files:
                if not name.endswith(".ipynb"):
                    continue
                path = current_path / name
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    errors.append(f"{_safe_path_label(path, root)}: file metadata unavailable ({type(exc).__name__})")
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    paths.append(path)
    except OSError as exc:
        onerror(exc)
    return sorted(paths), errors



def _notebook_check(root: Path) -> dict[str, str]:
    paths, scan_errors = discover_notebooks(root)
    if not paths and not scan_errors:
        return _result("Notebook JSON", "WARN", "no notebooks found")
    failures = list(scan_errors)
    for path in paths:
        try:
            if path.stat().st_size > MAX_NOTEBOOK_BYTES:
                failures.append(f"{_safe_path_label(path, root)}: notebook exceeds the 64 MiB inspection limit")
                continue
            with path.open("rb") as stream:
                raw = stream.read(MAX_NOTEBOOK_BYTES + 1)
            if len(raw) > MAX_NOTEBOOK_BYTES:
                failures.append(f"{_safe_path_label(path, root)}: notebook exceeds the 64 MiB inspection limit")
                continue
            data = json.loads(raw.decode("utf-8"))
            error = validate_notebook_document(data)
            if error:
                failures.append(f"{_safe_path_label(path, root)}: {error}")
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            failures.append(f"{_safe_path_label(path, root)}: could not be read ({type(exc).__name__})")
    if failures:
        return _result("Notebook JSON", "FAIL", f"{len(failures)} issue(s) across {len(paths)} notebook(s)", "\n".join(failures))
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
    try:
        files = sorted(root.glob("config.example*.json"))
    except OSError as exc:
        return _result(
            "Example config JSON",
            "FAIL",
            f"could not scan root config files ({type(exc).__name__})",
        )
    if not files:
        return _result("Example config JSON", "WARN", "no root config.example*.json found")
    errors = []
    for path in files:
        if _has_symlink_component(path, root):
            errors.append(f"{_safe_path_label(path, root)}: config file has a symlinked path component")
            continue
        try:
            if path.stat().st_size > MAX_CONFIG_BYTES:
                errors.append(f"{_safe_path_label(path, root)}: config exceeds the 1 MiB inspection limit")
                continue
            with path.open("rb") as stream:
                raw = stream.read(MAX_CONFIG_BYTES + 1)
            if len(raw) > MAX_CONFIG_BYTES:
                errors.append(f"{_safe_path_label(path, root)}: config exceeds the 1 MiB inspection limit")
                continue
            document = json.loads(raw.decode("utf-8"))
            error = validate_example_config(document)
            if error:
                errors.append(f"{_safe_path_label(path, root)}: {error}")
                continue
            detection = document.get("object_detection", {})
            labels_path = detection.get("labels_path") if isinstance(detection, dict) else None
            if labels_path:
                candidate = root / labels_path
                if _has_symlink_component(candidate, root):
                    errors.append(f"{_safe_path_label(path, root)}: object_detection.labels_path has a symlinked path component")
                    continue
                label_file = candidate.resolve()
                if not label_file.is_relative_to(root.resolve()):
                    errors.append(f"{_safe_path_label(path, root)}: object_detection.labels_path escapes the project root")
                    continue
                try:
                    if label_file.stat().st_size > MAX_CONFIG_BYTES:
                        errors.append(f"{_safe_path_label(path, root)}: object_detection.labels_path exceeds the 1 MiB inspection limit")
                        continue
                    with label_file.open("rb") as stream:
                        label_raw = stream.read(MAX_CONFIG_BYTES + 1)
                    if len(label_raw) > MAX_CONFIG_BYTES:
                        errors.append(f"{_safe_path_label(path, root)}: object_detection.labels_path exceeds the 1 MiB inspection limit")
                        continue
                    label_document = json.loads(label_raw.decode("utf-8"))
                    label_error = validate_label_map(label_document) if detection.get("backend") == "yolov8_onnx" else None
                    if label_error:
                        errors.append(f"{_safe_path_label(path, root)}: {label_error}")
                except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                    errors.append(f"{_safe_path_label(path, root)}: object_detection.labels_path could not be read ({type(exc).__name__})")
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            errors.append(f"{_safe_path_label(path, root)}: could not be read ({type(exc).__name__})")
    if errors:
        return _result("Example config JSON", "FAIL", f"{len(errors)} invalid file(s)", "\n".join(errors))
    return _result("Example config JSON", "PASS", f"{len(files)} example config file(s) and configured YOLO label maps validated")


MAX_IMPORTED_MANIFEST_BYTES = 16 * 1024 * 1024


def discover_import_manifests(imported: Path, root: Path) -> tuple[list[Path], list[str]]:
    """Find import manifests without following symlink directories; retain scan errors."""
    manifests: list[Path] = []
    errors: list[str] = []

    def onerror(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        try:
            relative = _safe_path_label(filename, root) if filename else "<unknown>"
        except (OSError, RuntimeError, ValueError, TypeError):
            relative = "<unavailable path>"
        errors.append(f"{relative}: directory scan failed ({type(error).__name__})")

    try:
        imported_info = imported.stat()
    except FileNotFoundError:
        return manifests, errors
    except OSError as exc:
        errors.append(f"models/imported: directory access failed ({type(exc).__name__})")
        return manifests, errors
    if not stat.S_ISDIR(imported_info.st_mode):
        errors.append("models/imported: expected a directory")
        return manifests, errors

    try:
        for current, directories, files in os.walk(
            imported, topdown=True, followlinks=False, onerror=onerror
        ):
            current_path = Path(current)
            directories[:] = [
                name for name in directories if not (current_path / name).is_symlink()
            ]
            if "import_manifest.json" in files:
                candidate = current_path / "import_manifest.json"
                if not _has_symlink_component(candidate, root):
                    manifests.append(candidate)
                else:
                    errors.append(f"{_safe_path_label(candidate.parent, root)}: symlinked manifest path")
    except OSError as exc:
        onerror(exc)
    return sorted(manifests), errors


def _model_integrity_check(root: Path) -> list[dict[str, str]]:
    imported = root / "models" / "imported"
    if _has_symlink_component(imported, root):
        return [_result("Imported model integrity", "FAIL",
                        "models/imported has a symlinked path component; refusing external content")]
    manifests, scan_errors = discover_import_manifests(imported, root)
    if scan_errors:
        results = [_result("Imported model integrity", "FAIL",
                           f"{len(scan_errors)} directory scan/path issue(s)", "\n".join(scan_errors))]
    else:
        results = []
    if not manifests and not scan_errors:
        return [_result("Imported model integrity", "WARN", "no imported model manifests found")]
    verifier = root / "scripts" / "verify_model_integrity.py"
    for manifest in manifests:
        relative = _safe_path_label(manifest.parent, root)
        if _has_symlink_component(manifest, root):
            results.append(_result("Imported model integrity", "FAIL",
                                   f"{relative} contains a symlinked path component"))
            continue
        try:
            if not manifest.is_file():
                raise OSError("manifest is not a regular file")
            if manifest.stat().st_size > MAX_IMPORTED_MANIFEST_BYTES:
                results.append(_result("Imported model integrity", "FAIL",
                                       f"{relative} manifest exceeds the 16 MiB inspection limit"))
                continue
            with manifest.open("rb") as stream:
                raw = stream.read(MAX_IMPORTED_MANIFEST_BYTES + 1)
            if len(raw) > MAX_IMPORTED_MANIFEST_BYTES:
                results.append(_result("Imported model integrity", "FAIL",
                                       f"{relative} manifest exceeds the 16 MiB inspection limit"))
                continue
            document = json.loads(raw.decode("utf-8"))
            integrity = document.get("integrity", {}) if isinstance(document, dict) else {}
            if not isinstance(integrity, dict) or not integrity.get("files"):
                results.append(_result("Imported model integrity", "WARN",
                                       f"{relative} has no SHA-256 inventory"))
                continue
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            results.append(_result("Imported model integrity", "FAIL",
                                   f"{relative} manifest unreadable ({type(exc).__name__})"))
            continue
        results.append(_run(
            f"Model SHA-256: {relative}",
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
        return _result("Optional Tesseract", "WARN", f"language discovery failed ({type(exc).__name__})")
    if proc.returncode:
        return _result("Optional Tesseract", "WARN", "language discovery failed", _redact_local_paths(proc.stdout or "", Path.cwd()))
    langs = [line.strip() for line in (proc.stdout or "").splitlines()
             if line.strip() and not line.lower().startswith("list of available")]
    return _result("Optional Tesseract", "INFO", "languages: " + (", ".join(langs) if langs else "none reported"))


def _write_report_pair(json_path: Path, md_path: Path, payload: dict[str, Any], markdown: str) -> None:
    """Write a report pair exclusively; roll back only files created here."""
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


def _write_reports(root: Path, results: list[dict[str, str]]) -> tuple[Path, Path]:
    report_parent = root / "reports" / "validation"
    if _has_symlink_component(report_parent, root):
        raise OSError("validation report path contains a symbolic link")
    report_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    base = report_parent / stamp
    target = base
    suffix = 0
    while True:
        if _has_symlink_component(target, root):
            raise OSError("validation report directory contains a symbolic link")
        try:
            target.mkdir()
            break
        except FileExistsError:
            suffix += 1
            target = base.with_name(f"{base.name}-{suffix}")
    counts = {s.lower(): sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "WARN", "INFO")}
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {"project": "NOORIX-AI", "generated_at": generated, "python": sys.version,
               "platform": platform.platform(), "results": results, "summary": counts}
    json_path, md_path = target / "validation-report.json", target / "validation-report.md"
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
    _write_report_pair(json_path, md_path, payload, "\n".join(lines))
    return json_path, md_path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = [_result(
        "Python version",
        "PASS" if python_version_supported(sys.version_info) else "FAIL",
        f"{sys.version.split()[0]} (Python 3.11–3.13 currently supported)",
    )]
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
    results.append(_run(
        "Declared dependency audit",
        [sys.executable, str(root / "scripts" / "dependency_audit.py")],
        root,
    ))
    results.append(_run(
        "Installed package capability check",
        [sys.executable, "-m", "noorix", "--check"],
        root,
    ))
    results.extend([_config_check(root), _notebook_check(root), *_model_integrity_check(root), _offline_bundle_check(root), _tesseract_check()])
    try:
        json_path, md_path = _write_reports(root, results)
    except OSError as exc:
        print(f"Could not write validation reports ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print("\nNOORIX-AI LOCAL VALIDATION")
    print("=" * 68)
    for item in results:
        print(f"[{item['status']:<4}] {item['name']}: {item['summary']}")
    failures = sum(item["status"] == "FAIL" for item in results)
    warnings = sum(item["status"] == "WARN" for item in results)
    print("-" * 68)
    print(f"FAIL: {failures} | WARN: {warnings}")
    print(f"JSON report: {_safe_path_label(json_path, root)}")
    print(f"Markdown report: {_safe_path_label(md_path, root)}")
    print("Automated PASS results are not production-readiness certification.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
