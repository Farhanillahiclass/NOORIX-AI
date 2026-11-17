"""Capability readiness declarations.

Statuses distinguish implemented adapters from missing model weights and
external executables. An adapter being present does not mean it is validated.
"""
from __future__ import annotations

from importlib.util import find_spec


def capability_status() -> dict[str, str]:
    """Report software readiness without opening the webcam or loading models."""
    return {
        "camera": "opencv_installed_device_unverified" if find_spec("cv2") else "install_camera_extra",
        "frame_quality": "available" if find_spec("cv2") and find_spec("numpy") else "install_camera_extra",
        "voice_command_parser": "text_intent_parser_available",
        "microphone_speech_to_text": (
            "file_transcription_adapter_available_microphone_capture_not_implemented"
            if find_spec("transformers") and find_spec("torch")
            else "install_asr_whisper_extra_file_transcription_only"
        ),
        "pakistani_currency_estimation": (
            "pytorch_mobilenet_v2_adapter_available_model_and_labels_required"
            if find_spec("torch") and find_spec("torchvision")
            else "install_pytorch_currency_extra"
        ),
        "everyday_object_detection": (
            "ssd_and_yolov8_onnx_adapters_available_model_and_labels_required"
            if find_spec("cv2") else "install_camera_extra"
        ),
        "continuous_live_detection": (
            "background_inference_available_camera_and_compatible_detector_required"
        ),
        "printed_urdu_ocr": "printed_ocr_adapter_available_tesseract_urd_data_required",
        "handwriting_ocr": "not_implemented_no_verified_model_adapter",
        "printed_english_ocr": (
            "adapter_available_tesseract_executable_required"
            if find_spec("pytesseract") else "install_ocr_extra_and_tesseract"
        ),
        "spoken_feedback": (
            "adapter_available_local_voice_unverified"
            if find_spec("pyttsx3") else "install_speech_extra"
        ),
    }
