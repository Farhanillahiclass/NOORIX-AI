from noorix.capabilities import capability_status


def test_models_are_not_claimed_ready_in_initial_scaffold():
    status = capability_status()
    assert status["pakistani_currency_estimation"] == "model_not_integrated"
    assert status["everyday_object_detection"] == "model_not_integrated"
    assert status["printed_english_ocr"] == "backend_not_integrated"
