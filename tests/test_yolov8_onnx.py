import json
import numpy as np
import pytest

from noorix.yolov8_onnx import YOLOv8ONNXDetector


def test_yolov8_requires_labels_and_valid_threshold():
    with pytest.raises(ValueError, match="labels"):
        YOLOv8ONNXDetector("missing.onnx", [])
    with pytest.raises(ValueError, match="confidence_threshold"):
        YOLOv8ONNXDetector("missing.onnx", ["person"], 1.1)


def test_yolov8_reads_list_and_numeric_mapping_labels(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(["person", "car"]), encoding="utf-8")
    assert YOLOv8ONNXDetector.read_labels(path) == ["person", "car"]
    path.write_text(json.dumps({"1": "car", "0": "person"}), encoding="utf-8")
    assert YOLOv8ONNXDetector.read_labels(path) == ["person", "car"]


def test_yolov8_rejects_non_contiguous_or_invalid_labels(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"0": "person", "2": "car"}), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        YOLOv8ONNXDetector.read_labels(path)
    path.write_text(json.dumps(["person", None]), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        YOLOv8ONNXDetector.read_labels(path)


def test_yolov8_requires_model_file(tmp_path):
    detector = YOLOv8ONNXDetector(tmp_path / "missing.onnx", ["person"])
    with pytest.raises(FileNotFoundError, match="YOLOv8 ONNX model"):
        detector.load()


def test_yolov8_decodes_standard_channel_first_output_and_letterbox():
    cv2 = pytest.importorskip("cv2")

    class FakeNet:
        def getUnconnectedOutLayersNames(self):
            return ["output"]
        def setInput(self, blob):
            self.blob = blob
        def forward(self, names):
            assert names == ["output"]
            output = np.zeros((1, 6, 2), dtype=np.float32)
            output[0, :, 0] = [320, 320, 100, 100, 0.95, 0.05]
            output[0, :, 1] = [320, 320, 100, 100, 0.90, 0.10]
            return output

    detector = YOLOv8ONNXDetector("unused.onnx", ["person", "car"], 0.3)
    detector._net = FakeNet()
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert len(result) == 1
    assert result[0]["label"] == "person"
    assert result[0]["confidence"] == pytest.approx(0.95)
    assert result[0]["position"] == "centre"
    assert result[0]["box"] == [270, 110, 370, 210]


def test_yolov8_rejects_unknown_output_contract():
    detector = YOLOv8ONNXDetector("unused.onnx", ["person"])
    with pytest.raises(ValueError, match="Unsupported YOLOv8 output shape"):
        detector._decode(np.zeros((1, 85, 8400)), 1.0, 0, 0, 640, 640)


def test_yolov8_rejects_none_and_invalid_images():
    detector = YOLOv8ONNXDetector("unused.onnx", ["person"])
    with pytest.raises(ValueError, match="must not be None"):
        detector.detect(None)
    with pytest.raises(ValueError, match="three-channel"):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
