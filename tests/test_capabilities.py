from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from noorix.capabilities import capability_status


def test_capabilities_resolve_configured_external_model_paths(tmp_path, monkeypatch):
    import noorix.capabilities as capabilities

    model_root = tmp_path / "external-models"
    currency = model_root / "currency"
    objects = model_root / "objects"
    asr_model = model_root / "asr" / "model"
    asr_processor = model_root / "asr" / "processor"
    for directory in (currency, objects, asr_model, asr_processor):
        directory.mkdir(parents=True)
    (currency / "weights.pth").write_bytes(b"weights")
    (currency / "labels.json").write_text(json.dumps({"classes": ["10_front"]}), encoding="utf-8")
    (objects / "detector.onnx").write_bytes(b"onnx")
    (objects / "labels.json").write_text(json.dumps({"0": "person"}), encoding="utf-8")
    (asr_model / "model.safetensors").write_bytes(b"weights")
    (asr_model / "config.json").write_text("{}", encoding="utf-8")
    (asr_processor / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "currency": {"backend": "pytorch_mobilenet_v2", "model_path": str(currency / "weights.pth"),
                     "labels_path": str(currency / "labels.json")},
        "object_detection": {"backend": "yolov8_onnx", "model_path": str(objects / "detector.onnx"),
                             "labels_path": str(objects / "labels.json")},
        "asr": {"model_path": str(asr_model), "processor_path": str(asr_processor)},
    }), encoding="utf-8")

    monkeypatch.setattr(capabilities, "find_spec", lambda name: object())
    fake_tesseract = SimpleNamespace(get_languages=lambda config="": ["eng", "urd", "osd"])
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)

    status = capability_status(config)
    assert status["pakistani_currency_estimation"] == "model_and_labels_present_not_inference_validated"
    assert status["everyday_object_detection"] == "model_and_labels_present_not_inference_validated"
    assert status["printed_english_ocr"] == "available_not_inference_validated"
    assert status["printed_urdu_ocr"] == "available_not_inference_validated"


def test_capabilities_report_missing_model_assets(tmp_path, monkeypatch):
    import noorix.capabilities as capabilities

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "currency": {"model_path": "missing.pth", "labels_path": "missing.json"},
        "object_detection": {"backend": "yolov8_onnx", "model_path": "missing.onnx",
                             "labels_path": "missing-labels.json"},
    }), encoding="utf-8")
    monkeypatch.setattr(capabilities, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "pytesseract", SimpleNamespace(
        get_languages=lambda config="": ["eng", "urd", "osd"]
    ))
    status = capability_status(config)
    assert status["pakistani_currency_estimation"] == "model_file_missing"
    assert status["everyday_object_detection"] == "model_file_missing"


def test_asr_is_not_misreported_as_live_microphone_capture(tmp_path, monkeypatch):
    import noorix.capabilities as capabilities
    monkeypatch.setattr(capabilities, "find_spec", lambda name: object())
    status = capability_status(tmp_path / "missing.json")
    assert "microphone_capture_not_implemented" in status["microphone_speech_to_text"]


def test_capability_status_reports_all_prototype_modules(tmp_path, monkeypatch):
    import noorix.capabilities as capabilities
    monkeypatch.setattr(capabilities, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "pytesseract", SimpleNamespace(
        get_languages=lambda config="": ["eng", "urd", "osd"]
    ))
    status = capability_status(tmp_path / "missing.json")
    assert set(status) == {
        "camera", "frame_quality", "voice_command_parser",
        "microphone_speech_to_text", "pakistani_currency_estimation",
        "everyday_object_detection", "continuous_live_detection",
        "printed_urdu_ocr", "handwriting_ocr", "printed_english_ocr",
        "spoken_feedback", "tesseract_executable",
    }
