"""Capability readiness declarations.

Statuses distinguish implemented adapters from missing model weights and
external executables. An adapter being present does not mean it is validated.
"""
from __future__ import annotations

from importlib.util import find_spec


def _module_available(name: str) -> bool:
    """Check optional-module availability without making health checks crash."""
    try:
        return find_spec(name) is not None
    except Exception:
        # Broken import metadata is a missing/unverified capability, not proof
        # that the module is installed and usable.
        return False


def capability_status() -> dict[str, str]:
    """Report software readiness without opening the webcam or loading models."""
    camera = _module_available("cv2")
    numpy = _module_available("numpy")
    return {
        "camera": "opencv_installed_device_unverified" if camera else "install_camera_extra",
        "frame_quality": "available" if camera and numpy else "install_camera_extra",
        "voice_command_parser": "text_intent_parser_available",
        "microphone_speech_to_text": "not_implemented_yet",
        "pakistani_currency_estimation": "onnx_model_and_validated_labels_required",
        "everyday_object_detection": "ssd_onnx_model_and_labels_required",
        "printed_english_ocr": (
            "adapter_available_tesseract_executable_required"
            if _module_available("pytesseract") else "install_ocr_extra_and_tesseract"
        ),
        "spoken_feedback": (
            "adapter_available_local_voice_unverified"
            if _module_available("pyttsx3") else "install_speech_extra"
        ),
    }
