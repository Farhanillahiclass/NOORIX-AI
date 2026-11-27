import json
from pathlib import Path

import numpy as np
import pytest

from noorix.ssd_mobilenet_v2 import SSDMobileNetV2Detector
from noorix.objects import create_object_detector


ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = ROOT / "models" / "object_detection" / "coco_labels.json"


def valid_labels():
    return [f"class_{index}" for index in range(90)]


def test_official_coco_labels_have_90_positions_and_80_names():
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    assert len(labels) == 90
    assert sum(label is not None for label in labels) == 80


def test_detector_accepts_unmapped_label_positions():
    labels = valid_labels()
    labels[11] = None

    detector = SSDMobileNetV2Detector("model.onnx", labels)

    assert len(detector.labels) == 90
    assert detector.labels[11] is None


def test_detector_rejects_incorrect_label_count():
    with pytest.raises(ValueError, match="exactly 90"):
        SSDMobileNetV2Detector("model.onnx", ["only_one"])


@pytest.mark.parametrize("invalid_label", ["", "   ", 123, [], {}])
def test_detector_rejects_invalid_label_values(invalid_label):
    labels = valid_labels()
    labels[0] = invalid_label

    with pytest.raises(ValueError, match="non-empty string or None"):
        SSDMobileNetV2Detector("model.onnx", labels)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_detector_rejects_invalid_confidence_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        SSDMobileNetV2Detector("model.onnx", valid_labels(), threshold)


def test_unknown_class_uses_safe_fallback_label(monkeypatch):
    import cv2

    labels = valid_labels()
    labels[11] = None

    detector = SSDMobileNetV2Detector("model.onnx", labels)
    detector._session = object()

    class FakeInput:
        name = "images"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, *args, **kwargs):
            return []

    detector._session = FakeSession()

    monkeypatch.setattr(
        detector,
        "_decode_heads",
        lambda outputs: (None, None),
    )
    monkeypatch.setattr(
        detector,
        "_postprocess",
        lambda boxes, logits: [
            (0.91, 11, np.array([0.1, 0.2, 0.8, 0.7], dtype=np.float32))
        ],
    )

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    results = detector.detect(image)

    assert len(results) == 1
    assert results[0]["label"] == "unknown_class_12"
    assert results[0]["class_id"] == 12
    assert results[0]["confidence"] == pytest.approx(0.91)
    assert len(results[0]["box"]) == 4


def test_ssd_factory_resolves_model_labels_and_anchor_paths(tmp_path):
    config_file = tmp_path / "config.json"
    labels_file = tmp_path / "labels.json"
    labels = valid_labels()

    labels_file.write_text(json.dumps(labels), encoding="utf-8")

    config = {
        "backend": "ssd_mobilenet_v2",
        "model_path": "models/model.onnx",
        "labels_path": "labels.json",
        "anchors_path": "models/anchors.npz",
    }

    detector = create_object_detector(config, config_file)

    assert isinstance(detector, SSDMobileNetV2Detector)
    assert detector.model_path == (tmp_path / "models" / "model.onnx").resolve()
    assert detector.anchors_path == (tmp_path / "models" / "anchors.npz").resolve()
    assert detector.labels == labels
    assert detector.confidence_threshold == pytest.approx(0.30)


def test_ssd_factory_rejects_missing_labels(tmp_path):
    config = {
        "backend": "ssd_mobilenet_v2",
        "model_path": "model.onnx",
    }

    with pytest.raises(ValueError, match="90-position labels list"):
        create_object_detector(config, tmp_path / "config.json")


def test_opencv_factory_remains_default_backend(tmp_path):
    config = {
        "model_path": "model.onnx",
        "labels": ["cup", "book"],
    }

    detector = create_object_detector(config, tmp_path / "config.json")

    assert detector.model_path == (tmp_path / "model.onnx").resolve()
    assert detector.labels == ["cup", "book"]
    assert detector.confidence_threshold == pytest.approx(0.50)


def test_factory_rejects_unsupported_backend(tmp_path):
    config = {
        "backend": "unknown_backend",
        "model_path": "model.onnx",
    }

    with pytest.raises(ValueError, match="Unsupported object detection backend"):
        create_object_detector(config, tmp_path / "config.json")



def test_ssd_load_rejects_wrong_output_metadata_without_retaining_session(tmp_path, monkeypatch):
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
            return [Tensor("images", [1, 3, 300, 300])]

        def get_outputs(self):
            values = []
            for index, (size, count) in enumerate(zip(
                SSDMobileNetV2Detector.HEAD_SIZES,
                SSDMobileNetV2Detector.HEAD_ANCHORS,
            )):
                values.append(Tensor(f"boxes_{index}", [1, count * 4, size, size]))
                values.append(Tensor(f"scores_{index}", [1, count * 91, size, size]))
            values[0].shape = [1, 12, 19, 19]
            return values

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    anchors = tmp_path / "anchors.npz"
    np.savez(anchors, anchors=np.zeros((1917, 4), dtype=np.float32))
    detector = SSDMobileNetV2Detector(model, valid_labels(), anchors_path=anchors)

    with pytest.raises(ValueError, match="Unexpected SSD output 0"):
        detector.load()

    assert detector._session is None
    assert detector._anchors is None


def test_ssd_load_rejects_non_float32_input_without_retaining_session(tmp_path, monkeypatch):
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
            return [Tensor("images", [1, 3, 300, 300], "tensor(float16)")]

        def get_outputs(self):
            return []

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    anchors = tmp_path / "anchors.npz"
    np.savez(anchors, anchors=np.zeros((1917, 4), dtype=np.float32))
    detector = SSDMobileNetV2Detector(model, valid_labels(), anchors_path=anchors)

    with pytest.raises(ValueError, match="input must be float32"):
        detector.load()

    assert detector._session is None
    assert detector._anchors is None


def _valid_raw_ssd_outputs():
    outputs = []
    for size, anchor_count in zip(
        SSDMobileNetV2Detector.HEAD_SIZES,
        SSDMobileNetV2Detector.HEAD_ANCHORS,
    ):
        outputs.append(np.zeros((1, anchor_count * 4, size, size), dtype=np.float32))
        outputs.append(np.zeros((1, anchor_count * 91, size, size), dtype=np.float32))
    return outputs


@pytest.mark.parametrize("head_index,value,expected", [
    (0, np.nan, "SSD box head 0 contains non-finite values"),
    (1, np.inf, "SSD class head 0 contains non-finite values"),
])
def test_ssd_decode_heads_rejects_non_finite_model_outputs(head_index, value, expected):
    detector = SSDMobileNetV2Detector("model.onnx", valid_labels())
    outputs = _valid_raw_ssd_outputs()
    outputs[head_index].flat[0] = value

    with pytest.raises(ValueError, match=expected):
        detector._decode_heads(outputs)


def test_ssd_decode_boxes_rejects_non_finite_encoded_values():
    detector = SSDMobileNetV2Detector("model.onnx", valid_labels())
    detector._anchors = np.zeros((1917, 4), dtype=np.float32)
    encoded = np.zeros((1917, 4), dtype=np.float32)
    encoded[0, 0] = np.nan

    with pytest.raises(ValueError, match="encoded boxes contain non-finite"):
        detector._decode_boxes(encoded)


def test_ssd_decode_boxes_rejects_overflowing_box_exponents():
    detector = SSDMobileNetV2Detector("model.onnx", valid_labels())
    detector._anchors = np.zeros((1917, 4), dtype=np.float32)
    encoded = np.zeros((1917, 4), dtype=np.float32)
    encoded[0, 2] = 1_000_000.0

    with pytest.raises(ValueError, match="Decoded SSD boxes contain non-finite"):
        detector._decode_boxes(encoded)


@pytest.mark.parametrize("setting,value", [
    ("confidence_threshold", True),
    ("confidence_threshold", "0.3"),
    ("confidence_threshold", float("nan")),
])
def test_ssd_factory_rejects_coerced_or_non_finite_thresholds(tmp_path, setting, value):
    config = {
        "backend": "ssd_mobilenet_v2",
        "model_path": "model.onnx",
        "labels": valid_labels(),
        setting: value,
    }
    with pytest.raises(ValueError, match="finite number"):
        create_object_detector(config, tmp_path / "config.json")


@pytest.mark.parametrize("setting,value", [
    ("confidence_threshold", True),
    ("confidence_threshold", "0.5"),
    ("iou_threshold", False),
    ("iou_threshold", float("inf")),
    ("input_size", 640.5),
    ("input_size", True),
    ("max_detections", 10.5),
    ("max_detections", True),
])
def test_yolov8_factory_rejects_coerced_numeric_settings(tmp_path, setting, value):
    from noorix.yolov8_onnx import YOLOv8ONNXDetector

    config = {
        "backend": "yolov8_onnx",
        "model_path": "model.onnx",
        "labels": list(YOLOv8ONNXDetector.COCO_LABELS),
        setting: value,
    }
    with pytest.raises(ValueError):
        create_object_detector(config, tmp_path / "config.json")


@pytest.mark.parametrize("value", [[300, True], [300, 300.5], "300,300"])
def test_opencv_factory_rejects_invalid_input_dimensions(tmp_path, value):
    config = {
        "backend": "opencv_dnn",
        "model_path": "model.onnx",
        "labels": ["person"],
        "input_size": value,
    }
    with pytest.raises(ValueError, match="exactly two integers"):
        create_object_detector(config, tmp_path / "config.json")


def test_detector_factory_rejects_non_object_configuration(tmp_path):
    with pytest.raises(ValueError, match="configuration must be an object"):
        create_object_detector(["not", "a", "mapping"], tmp_path / "config.json")


@pytest.mark.parametrize("threshold", [True, "0.3", float("nan"), float("inf")])
def test_ssd_detector_rejects_coerced_or_non_finite_thresholds(threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        SSDMobileNetV2Detector("model.onnx", valid_labels(), threshold)
