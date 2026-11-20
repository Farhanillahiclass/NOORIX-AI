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
    home = tmp_path / "home"
    sample = f"failed at {project / 'tests' / 'test_example.py'}; user={home / 'private.txt'}"
    redacted = _redact_local_paths(sample, project)

    assert str(project) not in redacted
    assert str(home) not in redacted
    assert "<PROJECT>" in redacted
    assert "<HOME>" in redacted
