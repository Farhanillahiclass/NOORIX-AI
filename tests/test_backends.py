import pytest

from noorix.currency import CurrencyEstimator
from noorix.objects import ObjectDetector
from noorix.ocr import PrintedEnglishOCR
from noorix.speech import SpeechOutput


def test_currency_requires_labels():
    with pytest.raises(ValueError, match="labels"):
        CurrencyEstimator("missing.onnx", [])


def test_currency_requires_model_file_before_loading_runtime():
    estimator = CurrencyEstimator("missing.onnx", ["100", "500"])
    with pytest.raises(FileNotFoundError, match="Currency model not found"):
        estimator.load()


def test_object_detector_validates_labels_and_threshold():
    with pytest.raises(ValueError, match="labels"):
        ObjectDetector("missing.onnx", [])
    with pytest.raises(ValueError, match="confidence_threshold"):
        ObjectDetector("missing.onnx", ["person"], 1.5)


def test_object_detector_requires_model_file():
    detector = ObjectDetector("missing.onnx", ["person"])
    with pytest.raises(FileNotFoundError, match="Object detection model not found"):
        detector.load()


def test_ocr_is_limited_to_english_and_rejects_empty_image():
    with pytest.raises(ValueError, match="English"):
        PrintedEnglishOCR(language="urd")
    with pytest.raises(ValueError, match="must not be None"):
        PrintedEnglishOCR().read(None)


def test_speech_validates_text_and_settings():
    with pytest.raises(ValueError, match="non-empty"):
        SpeechOutput().speak(" ")
    with pytest.raises(ValueError, match="rate"):
        SpeechOutput(rate=0)
    with pytest.raises(ValueError, match="volume"):
        SpeechOutput(volume=2.0)
