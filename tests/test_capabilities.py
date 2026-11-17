from noorix.capabilities import capability_status


def test_model_capabilities_are_not_claimed_ready_without_weights():
    status = capability_status()
    assert status["pakistani_currency_estimation"] in {
        "pytorch_mobilenet_v2_adapter_available_model_and_labels_required",
        "install_pytorch_currency_extra",
    }
    assert status["everyday_object_detection"] in {
        "ssd_and_yolov8_onnx_adapters_available_model_and_labels_required",
        "install_camera_extra",
    }
    assert status["handwriting_ocr"] == "not_implemented_no_verified_model_adapter"


def test_asr_is_not_misreported_as_live_microphone_capture():
    status = capability_status()
    assert "microphone_capture_not_implemented" in status["microphone_speech_to_text"] or (
        status["microphone_speech_to_text"] == "install_asr_whisper_extra_file_transcription_only"
    )


def test_capability_status_reports_all_prototype_modules():
    status = capability_status()
    assert set(status) == {
        "camera",
        "frame_quality",
        "voice_command_parser",
        "microphone_speech_to_text",
        "pakistani_currency_estimation",
        "everyday_object_detection",
        "continuous_live_detection",
        "printed_urdu_ocr",
        "handwriting_ocr",
        "printed_english_ocr",
        "spoken_feedback",
    }
