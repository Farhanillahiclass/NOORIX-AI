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


def test_object_detector_decodes_postprocessed_seven_column_output():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    class FakeNet:
        def getUnconnectedOutLayersNames(self):
            return ["detections"]

        def setInput(self, blob):
            self.blob = blob

        def forward(self, names):
            assert names == ["detections"]
            return [np.array([[[[0, 0, 0.9, 0.1, 0.2, 0.5, 0.8]]]], dtype=np.float32)]

    detector = ObjectDetector("unused.onnx", ["person"], confidence_threshold=0.5)
    detector._net = FakeNet()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    results = detector.detect(image)

    assert len(results) == 1
    assert results[0]["label"] == "person"
    assert results[0]["confidence"] == pytest.approx(0.9)
    assert results[0]["position"] == "centre"
    assert results[0]["box"] == [20, 20, 100, 80]


def test_object_detector_rejects_raw_multi_head_ssd_outputs():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    class FakeNet:
        def getUnconnectedOutLayersNames(self):
            return ["boxes", "scores"]

        def setInput(self, blob):
            self.blob = blob

        def forward(self, names):
            return [
                np.zeros((1, 4, 10, 10), dtype=np.float32),
                np.zeros((1, 90, 10, 10), dtype=np.float32),
            ]

    detector = ObjectDetector("unused.onnx", ["person"])
    detector._net = FakeNet()

    with pytest.raises(NotImplementedError, match="model-specific SSD decoder"):
        detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))


def test_object_detector_ignores_invalid_boxes_and_non_finite_rows():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    class FakeNet:
        def getUnconnectedOutLayersNames(self):
            return ["detections"]

        def setInput(self, blob):
            self.blob = blob

        def forward(self, names):
            return [np.array([
                [0, 0, 0.99, 0.8, 0.2, 0.3, 0.9],
                [0, 0, np.nan, 0.1, 0.2, 0.5, 0.8],
                [0, 0, 0.99, -0.1, 0.2, 0.5, 0.8],
            ], dtype=np.float32)]

    detector = ObjectDetector("unused.onnx", ["person"])
    detector._net = FakeNet()

    assert detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)) == []
