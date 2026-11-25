import pytest

from noorix.currency import CurrencyEstimator
from noorix.objects import ObjectDetector
from noorix.ocr import PrintedEnglishOCR
from noorix.speech import SpeechOutput


def test_currency_requires_labels():
    with pytest.raises(ValueError, match="labels"):
        CurrencyEstimator("missing.onnx", [])
    with pytest.raises(ValueError, match="non-empty denomination strings"):
        CurrencyEstimator("missing.onnx", ["100", " "])
    with pytest.raises(ValueError, match="positive integer"):
        CurrencyEstimator("missing.onnx", ["100"], input_size=True)
    with pytest.raises(ValueError, match="positive integer"):
        CurrencyEstimator("missing.onnx", ["100"], input_size=224.5)


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


def test_ocr_supports_english_and_urdu_and_rejects_invalid_input():
    assert PrintedEnglishOCR(language="eng").language == "eng"
    assert PrintedEnglishOCR(language="urd").language == "urd"
    assert PrintedEnglishOCR(language="eng+urd").language == "eng+urd"
    with pytest.raises(ValueError):
        PrintedEnglishOCR(language="invalid")
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
            return [np.array([[[[0, 0, 0.9, 0.3, 0.2, 0.7, 0.8]]]], dtype=np.float32)]

    detector = ObjectDetector("unused.onnx", ["person"], confidence_threshold=0.5)
    detector._net = FakeNet()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    results = detector.detect(image)

    assert len(results) == 1
    assert results[0]["label"] == "person"
    assert results[0]["confidence"] == pytest.approx(0.9)
    assert results[0]["position"] == "centre"
    assert results[0]["box"] == [60, 20, 140, 80]


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



def test_detector_factory_rejects_duplicate_json_label_keys(tmp_path):
    from noorix.objects import create_object_detector

    labels = tmp_path / "labels.json"
    labels.write_text('{"name": "person", "name": "other"}', encoding="utf-8")
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match="duplicate object key"):
        create_object_detector(
            {
                "backend": "yolov8_onnx",
                "model_path": "missing.onnx",
                "labels_path": "labels.json",
            },
            config_path,
        )



def test_currency_rejects_multiple_output_tensors():
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")

    class Input:
        name = "image"

    class FakeSession:
        def get_inputs(self):
            return [Input()]

        def run(self, _names, _inputs):
            return [np.array([[0.1, 0.9]], dtype=np.float32), np.array([[1.0]])]

    estimator = CurrencyEstimator("unused.onnx", ["100", "500"])
    estimator._session = FakeSession()

    with pytest.raises(RuntimeError, match="exactly one score tensor"):
        estimator.predict(np.zeros((32, 32, 3), dtype=np.uint8))


def test_currency_rejects_non_finite_model_scores():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    class Input:
        name = "image"

    class FakeSession:
        def get_inputs(self):
            return [Input()]

        def run(self, _names, _inputs):
            return [np.array([[np.nan, 0.9]], dtype=np.float32)]

    estimator = CurrencyEstimator("unused.onnx", ["100", "500"])
    estimator._session = FakeSession()

    with pytest.raises(RuntimeError, match="non-finite scores"):
        estimator.predict(np.zeros((32, 32, 3), dtype=np.uint8))


def test_currency_rejects_wrong_model_output_count():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    class Input:
        name = "image"

    class FakeSession:
        def get_inputs(self):
            return [Input()]

        def run(self, _names, _inputs):
            return [np.array([[0.1, 0.9]], dtype=np.float32)]

    estimator = CurrencyEstimator("unused.onnx", ["100", "500", "1000"])
    estimator._session = FakeSession()

    with pytest.raises(RuntimeError, match="scores for 3 labels"):
        estimator.predict(np.zeros((32, 32, 3), dtype=np.uint8))



def test_currency_load_rejects_non_float32_input_without_retaining_session(tmp_path, monkeypatch):
    import sys

    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class Session:
        def __init__(self, _path, providers):
            pass

        def get_inputs(self):
            return [Tensor("image", [1, 3, 224, 224], "tensor(float16)")]

        def get_outputs(self):
            return [Tensor("scores", [1, 2])]

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime())
    model = tmp_path / "currency.onnx"
    model.write_bytes(b"placeholder")
    estimator = CurrencyEstimator(model, ["100", "500"])

    with pytest.raises(ValueError, match="input must be float32"):
        estimator.load()

    assert estimator._session is None


def test_currency_load_rejects_output_label_dimension_mismatch(tmp_path, monkeypatch):
    import sys

    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class Session:
        def __init__(self, _path, providers):
            pass

        def get_inputs(self):
            return [Tensor("image", [1, 3, 224, 224])]

        def get_outputs(self):
            return [Tensor("scores", [1, 3])]

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime())
    model = tmp_path / "currency.onnx"
    model.write_bytes(b"placeholder")
    estimator = CurrencyEstimator(model, ["100", "500"])

    with pytest.raises(ValueError, match="one currency score per label"):
        estimator.load()

    assert estimator._session is None


def test_currency_load_rejects_multiple_input_tensors(tmp_path, monkeypatch):
    import sys

    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class MultiInputSession:
        def __init__(self, _path, providers):
            pass

        def get_inputs(self):
            return [Tensor("image1", [1, 3, 224, 224]), Tensor("image2", [1, 3, 224, 224])]

        def get_outputs(self):
            return [Tensor("scores", [1, 2])]

    class Runtime:
        InferenceSession = MultiInputSession

    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime())
    model = tmp_path / "currency.onnx"
    model.write_bytes(b"placeholder")
    estimator = CurrencyEstimator(model, ["100", "500"])

    with pytest.raises(ValueError, match="exactly one input"):
        estimator.load()

    assert estimator._session is None


def test_currency_rejects_duplicate_denomination_labels():
    with pytest.raises(ValueError, match="unique after whitespace and case normalization"):
        CurrencyEstimator("missing.onnx", ["100", "100"])
    with pytest.raises(ValueError, match="unique after whitespace and case normalization"):
        CurrencyEstimator("missing.onnx", ["Rs 500", " rs 500 "])
