import numpy as np
import pytest

from noorix.yolov8_onnx import YOLOv8ONNXDetector


LABELS = list(YOLOv8ONNXDetector.COCO_LABELS)


def make_detector(**kwargs):
    return YOLOv8ONNXDetector("unused.onnx", LABELS, **kwargs)


def test_detector_requires_exactly_80_labels():
    with pytest.raises(ValueError, match="exactly 80"):
        YOLOv8ONNXDetector("unused.onnx", LABELS[:79])


def test_detector_validates_thresholds():
    with pytest.raises(ValueError, match="confidence_threshold"):
        make_detector(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="iou_threshold"):
        make_detector(iou_threshold=0)


def test_normalize_output_accepts_channel_first_and_candidate_first():
    detector = make_detector()
    channel_first = np.zeros((1, 84, 5), dtype=np.float32)
    candidate_first = np.zeros((1, 5, 84), dtype=np.float32)
    assert detector._normalize_output(channel_first).shape == (5, 84)
    assert detector._normalize_output(candidate_first).shape == (5, 84)


def test_normalize_output_rejects_incompatible_model_head():
    detector = make_detector()
    with pytest.raises(ValueError, match="84"):
        detector._normalize_output(np.zeros((1, 100, 6), dtype=np.float32))


def test_postprocess_returns_coco_label_and_original_image_box():
    detector = make_detector(confidence_threshold=0.2)
    # 640x640 input; one candidate centred at (320, 320), size 200x100.
    output = np.zeros((1, 84, 1), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 200, 100]
    output[0, 4 + 7, 0] = 0.91

    results = detector._postprocess(output, (640, 640), 1.0, 0, 0)
    assert len(results) == 1
    assert results[0]["label"] == "truck"
    assert results[0]["class_id"] == 7
    assert results[0]["confidence"] == pytest.approx(0.91)
    assert results[0]["box"] == [220, 270, 420, 370]


def test_postprocess_maps_letterboxed_coordinates_back_to_source():
    detector = make_detector(confidence_threshold=0.2)
    output = np.zeros((1, 84, 1), dtype=np.float32)
    # Original image 320x640 is letterboxed to 640x640: scale=1, pad_y=160.
    output[0, :4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = 0.95
    results = detector._postprocess(output, (320, 640), 1.0, 0, 160)
    assert results[0]["box"] == [270, 110, 370, 210]


def test_postprocess_suppresses_overlapping_boxes_of_same_class():
    detector = make_detector(confidence_threshold=0.2, iou_threshold=0.4)
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 200, 200]
    output[0, :4, 1] = [322, 322, 200, 200]
    output[0, 4, 0] = 0.9
    output[0, 4, 1] = 0.8
    assert len(detector._postprocess(output, (640, 640), 1.0, 0, 0)) == 1


def test_postprocess_keeps_overlapping_boxes_from_different_classes():
    detector = make_detector(confidence_threshold=0.2)
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 200, 200]
    output[0, :4, 1] = [322, 322, 200, 200]
    output[0, 4, 0] = 0.9
    output[0, 5, 1] = 0.8
    assert len(detector._postprocess(output, (640, 640), 1.0, 0, 0)) == 2


def test_postprocess_ignores_non_finite_and_low_confidence_candidates():
    detector = make_detector(confidence_threshold=0.5)
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = np.nan
    output[0, :4, 1] = [320, 320, 100, 100]
    output[0, 4, 1] = 0.1
    assert detector._postprocess(output, (640, 640), 1.0, 0, 0) == []



def test_detector_rejects_wrong_coco_label_order():
    labels = list(LABELS)
    labels[0], labels[1] = labels[1], labels[0]
    with pytest.raises(ValueError, match="canonical COCO-80"):
        YOLOv8ONNXDetector("unused.onnx", labels)


def test_detector_rejects_duplicate_and_blank_labels():
    labels = list(YOLOv8ONNXDetector.COCO_LABELS)
    labels[1] = labels[0]
    with pytest.raises(ValueError, match="unique"):
        YOLOv8ONNXDetector("unused.onnx", labels)
    labels = list(YOLOv8ONNXDetector.COCO_LABELS)
    labels[0] = " "
    with pytest.raises(ValueError, match="non-empty"):
        YOLOv8ONNXDetector("unused.onnx", labels)


def test_load_validates_output_metadata_before_inference(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class Session:
        def __init__(self, _path, providers):
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [Tensor("images", [1, 3, 640, 640])]

        def get_outputs(self):
            return [Tensor("output0", [1, 84, 8400])]

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", Runtime())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    detector = YOLOv8ONNXDetector(model, list(YOLOv8ONNXDetector.COCO_LABELS))
    detector.load()
    assert detector._input_name == "images"


def test_load_rejects_incompatible_output_metadata(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class Session:
        def __init__(self, _path, providers):
            pass

        def get_inputs(self):
            return [Tensor("images", [1, 3, 640, 640])]

        def get_outputs(self):
            return [Tensor("output0", [1, 100, 6])]

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", Runtime())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    detector = YOLOv8ONNXDetector(model, list(YOLOv8ONNXDetector.COCO_LABELS))
    with pytest.raises(ValueError, match="84-value feature axis"):
        detector.load()


def test_normalize_output_rejects_excessive_candidate_count():
    detector = make_detector()
    output = np.zeros((1, 84, detector.MAX_CANDIDATES + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="candidate safety limit"):
        detector._normalize_output(output)


def test_load_rejects_excessive_candidate_metadata(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, name, shape, tensor_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = tensor_type

    class Session:
        def __init__(self, _path, providers):
            pass

        def get_inputs(self):
            return [Tensor("images", [1, 3, 640, 640])]

        def get_outputs(self):
            return [Tensor("output0", [1, 84, 20_001])]

    class Runtime:
        InferenceSession = Session

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", Runtime())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    detector = YOLOv8ONNXDetector(model, list(YOLOv8ONNXDetector.COCO_LABELS))
    with pytest.raises(ValueError, match="candidate safety limit"):
        detector.load()
