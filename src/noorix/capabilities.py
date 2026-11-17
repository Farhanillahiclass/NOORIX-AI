"""Config-aware local readiness checks for NOORIX specialist adapters."""
from __future__ import annotations

import json
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from noorix.paths import resolve_config_path


def _read_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _labels_available(path_value: Any, config_path: str | Path, kind: str) -> bool:
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    path = resolve_config_path(path_value, config_path)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if kind == "currency":
        labels = data.get("classes") if isinstance(data, dict) else None
        return isinstance(labels, list) and bool(labels) and all(
            isinstance(item, str) and item.strip() for item in labels
        )
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        try:
            keys = sorted(data, key=lambda item: int(item))
            labels = [data[key] for key in keys]
            if [int(key) for key in keys] != list(range(len(keys))):
                return False
        except (TypeError, ValueError):
            return False
    else:
        return False
    return bool(labels) and all(isinstance(item, str) and item.strip() for item in labels)


def _tesseract_languages() -> tuple[bool, set[str], str | None]:
    if not find_spec("pytesseract"):
        return False, set(), "pytesseract_not_installed"
    try:
        import pytesseract
        languages = set(pytesseract.get_languages(config=""))
        return True, languages, None
    except Exception as exc:
        return False, set(), f"{type(exc).__name__}: {exc}"


def capability_status(config_path: str | Path = "config.json") -> dict[str, str]:
    """Inspect configured assets without loading models or opening the camera.

    A present model is deliberately reported as not inference-validated until
    an actual specialist smoke test has loaded and run it.
    """
    config_path = Path(config_path)
    config = _read_config(config_path)
    currency = config.get("currency", {})
    objects = config.get("object_detection", {})
    asr = config.get("asr", {})
    ocr = config.get("ocr", {})
    currency = currency if isinstance(currency, dict) else {}
    objects = objects if isinstance(objects, dict) else {}
    asr = asr if isinstance(asr, dict) else {}
    ocr = ocr if isinstance(ocr, dict) else {}

    torch_ready = bool(find_spec("torch") and find_spec("torchvision"))
    if not torch_ready:
        currency_status = "install_pytorch_currency_extra"
    elif not currency.get("model_path"):
        currency_status = "model_not_configured"
    elif not resolve_config_path(currency["model_path"], config_path).is_file():
        currency_status = "model_file_missing"
    elif not _labels_available(currency.get("labels_path"), config_path, "currency"):
        currency_status = "labels_missing_or_invalid"
    else:
        currency_status = "model_and_labels_present_not_inference_validated"

    if not find_spec("cv2"):
        object_status = "install_camera_extra"
    elif not objects.get("model_path"):
        object_status = "model_not_configured"
    elif not resolve_config_path(objects["model_path"], config_path).is_file():
        object_status = "model_file_missing"
    elif str(objects.get("backend", "")).lower() == "yolov8_onnx" and not _labels_available(
        objects.get("labels_path"), config_path, "objects"
    ):
        object_status = "labels_missing_or_invalid"
    elif str(objects.get("backend", "")).lower() == "ssd_mobilenet_v2" and not (
        objects.get("labels_path") or isinstance(objects.get("labels"), list)
    ):
        object_status = "labels_not_configured"
    else:
        object_status = "model_and_labels_present_not_inference_validated"

    tesseract_ok, languages, tesseract_error = _tesseract_languages()
    if not tesseract_ok:
        english_ocr = "tesseract_unavailable" if tesseract_error else "install_ocr_extra"
        urdu_ocr = english_ocr
    else:
        english_ocr = "available_not_inference_validated" if "eng" in languages else "eng_language_data_missing"
        urdu_ocr = "available_not_inference_validated" if "urd" in languages else "urd_language_data_missing"

    asr_dependencies = bool(find_spec("transformers") and find_spec("torch"))
    if not asr_dependencies:
        asr_status = "install_asr_whisper_extra_file_transcription_only"
    elif not asr.get("model_path"):
        asr_status = "model_not_configured"
    else:
        model_path = resolve_config_path(asr["model_path"], config_path)
        processor_value = asr.get("processor_path", asr["model_path"])
        processor_path = resolve_config_path(processor_value, config_path)
        if not model_path.is_dir():
            asr_status = "model_directory_missing"
        elif not processor_path.is_dir():
            asr_status = "processor_directory_missing"
        elif not ((model_path / "model.safetensors").is_file() or
                  (model_path / "pytorch_model.bin").is_file()):
            asr_status = "model_weights_missing"
        elif not (model_path / "config.json").is_file():
            asr_status = "model_config_missing"
        elif not (processor_path / "tokenizer.json").is_file():
            asr_status = "processor_tokenizer_missing"
        else:
            asr_status = "model_and_processor_present_not_inference_validated"

    return {
        "camera": "opencv_installed_device_unverified" if find_spec("cv2") else "install_camera_extra",
        "frame_quality": "available" if find_spec("cv2") and find_spec("numpy") else "install_camera_extra",
        "voice_command_parser": "text_intent_parser_available",
        "microphone_speech_to_text": (
            "file_transcription_adapter_available_microphone_capture_not_implemented"
            if asr_dependencies else "install_asr_whisper_extra_file_transcription_only"
        ),
        "pakistani_currency_estimation": currency_status,
        "everyday_object_detection": object_status,
        "continuous_live_detection": "background_inference_available_camera_and_compatible_detector_required",
        "printed_urdu_ocr": urdu_ocr,
        "handwriting_ocr": "not_implemented_no_verified_model_adapter",
        "printed_english_ocr": english_ocr,
        "spoken_feedback": (
            "adapter_available_local_voice_unverified"
            if find_spec("pyttsx3") else "install_speech_extra"
        ),
        "tesseract_executable": "available" if tesseract_ok else "not_found_or_unusable",
    }
