from scripts.run_full_validation import (
    python_version_supported,
    validate_example_config,
    validate_notebook_document,
)


def test_valid_empty_notebook_is_accepted():
    assert validate_notebook_document({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [],
    }) is None


def test_notebook_requires_object_root():
    assert validate_notebook_document([]) == "top-level JSON value must be an object"


def test_notebook_rejects_unsupported_format():
    assert validate_notebook_document({"nbformat": 3, "nbformat_minor": 0, "cells": []}) == "nbformat must be 4"


def test_notebook_rejects_invalid_cell_source():
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [{"cell_type": "markdown", "source": 42}],
    }
    assert validate_notebook_document(document) == "cell 0 source must be a string or list"


def test_code_cell_requires_list_outputs():
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [{"cell_type": "code", "source": "print('hi')", "outputs": {}}],
    }
    assert validate_notebook_document(document) == "code cell 0 outputs must be a list"


def test_valid_example_config_is_accepted():
    assert validate_example_config({
        "object_detection": {
            "backend": "yolov8_onnx",
            "model_path": "models/model.onnx",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "input_size": 640,
            "max_detections": 100,
        },
        "ocr": {"language": "eng+urd"},
        "currency": {"labels": ["note"], "input_size": 224},
    }) is None


def test_config_rejects_non_object_root():
    assert validate_example_config([]) == "configuration root must be an object"


def test_config_rejects_out_of_range_confidence():
    error = validate_example_config({"object_detection": {"confidence_threshold": 1.5}})
    assert error == "object_detection.confidence_threshold must be a number between 0 and 1"


def test_config_rejects_boolean_as_numeric_threshold():
    error = validate_example_config({"object_detection": {"iou_threshold": True}})
    assert error == "object_detection.iou_threshold must be a number between 0 and 1"


def test_config_rejects_non_positive_input_size():
    error = validate_example_config({"object_detection": {"input_size": 0}})
    assert error == "object_detection.input_size must be a positive integer"


def test_config_rejects_unsupported_ocr_language():
    error = validate_example_config({"ocr": {"language": "auto"}})
    assert error == "ocr.language must be eng, urd, or eng+urd"


def test_config_rejects_invalid_currency_labels():
    error = validate_example_config({"currency": {"labels": "one,two"}})
    assert error == "currency.labels must be a list of strings"


def test_config_rejects_non_string_ocr_language_without_crashing():
    error = validate_example_config({"ocr": {"language": ["eng"]}})
    assert error == "ocr.language must be eng, urd, or eng+urd"


from scripts.run_full_validation import validate_label_map


def test_valid_coco_label_map_is_accepted():
    labels = [f"label-{index}" for index in range(80)]
    assert validate_label_map(labels) is None


def test_label_map_requires_exactly_80_entries():
    assert "exactly 80" in validate_label_map(["person"])


def test_label_map_rejects_blank_and_duplicate_entries():
    labels = [f"label-{index}" for index in range(80)]
    labels[1] = ""
    assert validate_label_map(labels) == "label map entries must be non-empty strings"
    labels[1] = labels[0]
    assert validate_label_map(labels) == "label map entries must be unique"


def test_label_map_requires_json_list():
    assert validate_label_map({"0": "person"}) == "label map root must be a JSON list"



def test_supported_python_range_matches_ci_matrix():
    assert python_version_supported((3, 11, 0))
    assert python_version_supported((3, 12, 9))
    assert python_version_supported((3, 13, 0))
    assert not python_version_supported((3, 10, 99))
    assert not python_version_supported((3, 14, 0))


def test_notebook_rejects_boolean_or_negative_minor_version():
    base = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
    for invalid in (True, -1):
        document = {**base, "nbformat_minor": invalid}
        assert validate_notebook_document(document) == "nbformat_minor must be a non-negative integer"


def test_notebook_rejects_boolean_or_negative_execution_count():
    for invalid in (True, -1):
        document = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{
                "cell_type": "code",
                "source": "print('hi')",
                "outputs": [],
                "execution_count": invalid,
            }],
        }
        assert validate_notebook_document(document) == "code cell 0 execution_count must be a non-negative integer or null"



def test_validation_report_redacts_project_and_home_paths(tmp_path):
    from scripts.run_full_validation import _redact_local_paths

    project = tmp_path / "project"
    home = __import__("pathlib").Path.home()
    sample = f"failed at {project / 'tests' / 'test_example.py'}; user={home / 'private.txt'}"
    redacted = _redact_local_paths(sample, project)

    assert str(project) not in redacted
    assert str(home) not in redacted
    assert "<PROJECT>" in redacted
    assert "<HOME>" in redacted



def test_validation_refuses_symlinked_input_components_when_supported(tmp_path):
    import pytest
    from scripts.run_full_validation import _has_symlink_component

    root = tmp_path / "project"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    alias = root / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    assert _has_symlink_component(alias / "manifest.json", root)
    assert not _has_symlink_component(real / "manifest.json", root)



def test_discover_notebooks_skips_symlinked_directories(tmp_path):
    import pytest
    from run_full_validation import discover_notebooks

    real = tmp_path / "real"
    real.mkdir()
    (real / "hidden.ipynb").write_text("{}", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    found, errors = discover_notebooks(tmp_path)
    assert found == [real / "hidden.ipynb"]
    assert errors == []


def test_discover_notebooks_reports_walk_errors(tmp_path, monkeypatch):
    import run_full_validation

    def failing_walk(*args, **kwargs):
        onerror = kwargs["onerror"]
        onerror(PermissionError("private path"))
        return iter(())

    monkeypatch.setattr(run_full_validation.os, "walk", failing_walk)
    found, errors = run_full_validation.discover_notebooks(tmp_path)
    assert found == []
    assert len(errors) == 1
    assert "PermissionError" in errors[0]
    assert "private path" not in errors[0]



def test_validation_report_rejects_symlinked_output_directory(tmp_path):
    import pytest
    from scripts.run_full_validation import _write_reports

    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external-reports"
    external.mkdir()
    (root / "reports").symlink_to(external, target_is_directory=True)

    with pytest.raises(OSError, match="symbolic link"):
        _write_reports(root, [])
    assert list(external.iterdir()) == []



def test_config_check_rejects_symlinked_example_config(tmp_path):
    import json
    import pytest
    from scripts.run_full_validation import _config_check

    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external-config.json"
    external.write_text(json.dumps({"ocr": {"language": "eng"}}), encoding="utf-8")
    link = root / "config.example.json"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "symlinked path component" in result["output"]


def test_config_check_rejects_symlinked_label_map(tmp_path):
    import json
    import pytest
    from scripts.run_full_validation import _config_check

    root = tmp_path / "project"
    (root / "models").mkdir(parents=True)
    external = tmp_path / "external-labels.json"
    external.write_text(json.dumps([f"label-{index}" for index in range(80)]), encoding="utf-8")
    link = root / "models" / "labels.json"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")
    (root / "config.example.json").write_text(json.dumps({
        "object_detection": {
            "backend": "yolov8_onnx",
            "labels_path": "models/labels.json",
        }
    }), encoding="utf-8")

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "labels_path has a symlinked path component" in result["output"]


def test_validation_report_uses_new_directory_when_timestamp_already_exists(tmp_path):
    from datetime import datetime
    from scripts.run_full_validation import _write_reports

    root = tmp_path / "project"
    parent = root / "reports" / "validation"
    parent.mkdir(parents=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    existing = parent / stamp
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    json_path, markdown_path = _write_reports(root, [])

    assert json_path.parent == parent / f"{stamp}-1"
    assert markdown_path.exists()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_validation_report_files_are_created_exclusively(tmp_path):
    from scripts.run_full_validation import _write_reports

    root = tmp_path / "project"
    root.mkdir()
    json_path, markdown_path = _write_reports(root, [])

    assert json_path.is_file()
    assert markdown_path.is_file()


def test_model_integrity_check_rejects_symlinked_models_parent(tmp_path):
    import pytest
    from scripts.run_full_validation import _model_integrity_check

    root = tmp_path / "project"
    external = tmp_path / "external"
    (external / "imported").mkdir(parents=True)
    (root).mkdir()
    try:
        (root / "models").symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    result = _model_integrity_check(root)
    assert result[0]["status"] == "FAIL"
    assert "symlinked path component" in result[0]["summary"]


def test_model_integrity_check_rejects_oversized_manifest_before_json_parse(tmp_path):
    from scripts.run_full_validation import MAX_IMPORTED_MANIFEST_BYTES, _model_integrity_check

    root = tmp_path / "project"
    package = root / "models" / "imported" / "sample"
    package.mkdir(parents=True)
    manifest = package / "import_manifest.json"
    manifest.write_bytes(b" " * (MAX_IMPORTED_MANIFEST_BYTES + 1))

    result = _model_integrity_check(root)
    assert result[0]["status"] == "FAIL"
    assert "16 MiB inspection limit" in result[0]["summary"]


def test_notebook_check_rejects_oversized_notebook_before_json_parse(tmp_path):
    from scripts.run_full_validation import MAX_NOTEBOOK_BYTES, _notebook_check

    root = tmp_path / "project"
    root.mkdir()
    notebook = root / "large.ipynb"
    notebook.write_bytes(b" " * (MAX_NOTEBOOK_BYTES + 1))

    result = _notebook_check(root)
    assert result["status"] == "FAIL"
    assert "64 MiB inspection limit" in result["output"]


def test_config_check_rejects_oversized_example_config(tmp_path):
    from scripts.run_full_validation import MAX_CONFIG_BYTES, _config_check

    root = tmp_path / "project"
    root.mkdir()
    (root / "config.example.json").write_bytes(b" " * (MAX_CONFIG_BYTES + 1))

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "1 MiB inspection limit" in result["output"]


def test_config_check_rejects_oversized_label_map(tmp_path):
    import json
    from scripts.run_full_validation import MAX_CONFIG_BYTES, _config_check

    root = tmp_path / "project"
    models = root / "models"
    models.mkdir(parents=True)
    labels = models / "labels.json"
    labels.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    (root / "config.example.json").write_text(json.dumps({
        "object_detection": {
            "backend": "yolov8_onnx",
            "labels_path": "models/labels.json",
        }
    }), encoding="utf-8")

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "labels_path exceeds the 1 MiB inspection limit" in result["output"]



def test_deeply_nested_notebook_json_is_reported_without_crashing(tmp_path):
    from scripts.run_full_validation import _notebook_check

    root = tmp_path / "project"
    root.mkdir()
    nested = "[" * 2000 + "0" + "]" * 2000
    (root / "deep.ipynb").write_text(nested, encoding="utf-8")

    result = _notebook_check(root)
    assert result["status"] == "FAIL"
    assert "RecursionError" in result["output"]


def test_deeply_nested_example_config_is_reported_without_crashing(tmp_path):
    from scripts.run_full_validation import _config_check

    root = tmp_path / "project"
    root.mkdir()
    nested = "[" * 2000 + "0" + "]" * 2000
    (root / "config.example.json").write_text(nested, encoding="utf-8")

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "RecursionError" in result["output"]


def test_deeply_nested_label_map_is_reported_without_crashing(tmp_path):
    import json
    from scripts.run_full_validation import _config_check

    root = tmp_path / "project"
    root.mkdir()
    (root / "config.example.json").write_text(json.dumps({
        "object_detection": {"backend": "yolov8_onnx", "labels_path": "labels.json"}
    }), encoding="utf-8")
    nested = "[" * 2000 + "0" + "]" * 2000
    (root / "labels.json").write_text(nested, encoding="utf-8")

    result = _config_check(root)
    assert result["status"] == "FAIL"
    assert "RecursionError" in result["output"]


def test_deeply_nested_import_manifest_is_reported_without_crashing(tmp_path):
    from scripts.run_full_validation import _model_integrity_check

    root = tmp_path / "project"
    imported = root / "models" / "imported" / "sample"
    imported.mkdir(parents=True)
    nested = "[" * 2000 + "0" + "]" * 2000
    (imported / "import_manifest.json").write_text(nested, encoding="utf-8")

    results = _model_integrity_check(root)
    assert len(results) == 1
    assert results[0]["status"] == "FAIL"
    assert "RecursionError" in results[0]["summary"]


def test_validation_report_pair_cleans_partial_json_on_markdown_failure(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    from scripts.run_full_validation import _write_report_pair

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path == markdown_path:
            raise OSError("simulated markdown write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="simulated markdown write failure"):
        _write_report_pair(json_path, markdown_path, {"ok": True}, "# report")

    assert not json_path.exists()
    assert not markdown_path.exists()
