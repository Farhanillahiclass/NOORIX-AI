from noorix.capabilities import capability_status


def test_models_are_not_claimed_ready_without_weights():
    status = capability_status()
    assert status["pakistani_currency_estimation"] == "onnx_model_and_validated_labels_required"
    assert status["everyday_object_detection"] == "ssd_onnx_model_and_labels_required"


def test_capability_status_reports_all_prototype_modules():
    status = capability_status()
    assert set(status) == {
        "camera",
        "frame_quality",
        "voice_command_parser",
        "microphone_speech_to_text",
        "pakistani_currency_estimation",
        "everyday_object_detection",
        "printed_english_ocr",
        "spoken_feedback",
    }
