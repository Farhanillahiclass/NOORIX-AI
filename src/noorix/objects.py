"""OpenCV DNN adapter for post-processed SSD ONNX detectors.

The adapter accepts one output tensor containing rows in the format
[image_id, class_id, confidence, x1, y1, x2, y2]. Raw multi-head SSD models
must be decoded by a model-specific, independently verified decoder first.
"""
from __future__ import annotations

from pathlib import Path

from noorix.position import image_relative_position


class ObjectDetector:
    """Run an SSD detector that emits a single, post-processed 7-column tensor.

    This deliberately does not guess anchor generation, box coding, class
    activation, or non-maximum suppression for raw SSD model outputs.
    """

    def __init__(
        self, model_path: str | Path, labels: list[str],
        confidence_threshold: float = 0.50, input_size: tuple[int, int] = (300, 300),
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self._net = None
        if not self.labels:
            raise ValueError("labels must not be empty")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if len(input_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in input_size):
            raise ValueError("input_size must contain two positive integers")

    def load(self) -> None:
        from noorix.paths import require_regular_file

        self.model_path = require_regular_file(self.model_path, label="Object detection model")
        import cv2
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def _forward_detections(self):
        """Return a single post-processed detection tensor or fail explicitly."""
        import numpy as np

        output_names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(output_names)
        if isinstance(outputs, (list, tuple)):
            if len(outputs) != 1:
                raise NotImplementedError(
                    "This model exposes multiple raw output tensors. "
                    "A model-specific SSD decoder is required; raw outputs "
                    "are not interchangeable with [image_id, class_id, "
                    "confidence, x1, y1, x2, y2] detections."
                )
            outputs = outputs[0]
        array = np.asarray(outputs)
        if array.size == 0:
            return array.reshape(0, 7)
        if array.size % 7:
            raise ValueError(
                "Object detector output must contain rows of exactly 7 values; "
                f"received output shape {array.shape}"
            )
        return array.reshape(-1, 7)

    def detect(self, image) -> list[dict[str, object]]:
        if image is None:
            raise ValueError("image must not be None")
        if self._net is None:
            self.load()
        import cv2
        import numpy as np

        if not hasattr(image, "shape") or len(image.shape) < 2:
            raise ValueError("image must have height and width dimensions")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be greater than zero")

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 127.5, size=self.input_size,
            mean=(127.5, 127.5, 127.5), swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        detections = self._forward_detections()
        results = []
        for row in detections:
            if not np.isfinite(row).all():
                continue
            class_value = float(row[1])
            confidence = float(row[2])
            if not class_value.is_integer():
                continue
            class_index = int(class_value)
            if confidence < self.confidence_threshold or not 0 <= class_index < len(self.labels):
                continue
            x1, y1, x2, y2 = (float(v) for v in row[3:7])
            if x2 <= x1 or y2 <= y1:
                continue
            # The supported post-processed contract uses normalized coordinates.
            if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0
                    and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                continue
            centre_x = ((x1 + x2) / 2.0) * width
            # Round float32 normalized coordinates to avoid truncation from
            # representation error (for example, 0.7 * 200 becoming 139).
            results.append({
                "label": self.labels[class_index],
                "confidence": confidence,
                "position": image_relative_position(centre_x, width),
                "box": [
                    max(0, min(width, round(x1 * width))),
                    max(0, min(height, round(y1 * height))),
                    max(0, min(width, round(x2 * width))),
                    max(0, min(height, round(y2 * height))),
                ],
            })
        return results

def create_object_detector(config: dict, config_path: str | Path = "config.json"):
    """Create the configured object detector using paths relative to config."""
    from noorix.paths import load_json_document, resolve_config_path

    backend = str(config.get("backend", "opencv_dnn")).lower()
    model_value = config.get("model_path")
    if not model_value:
        raise ValueError("Object detection model_path is not configured")

    model_path = resolve_config_path(model_value, config_path)
    threshold = float(config.get(
        "confidence_threshold",
        0.30 if backend == "ssd_mobilenet_v2" else 0.50,
    ))

    if backend == "ssd_mobilenet_v2":
        from noorix.ssd_mobilenet_v2 import SSDMobileNetV2Detector

        labels = config.get("labels")
        labels_path = config.get("labels_path")

        if labels_path:
            resolved_labels = resolve_config_path(labels_path, config_path)
            labels = load_json_document(resolved_labels)

        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError(
                "SSD MobileNet V2 requires labels_path or a 90-position labels list"
            )

        anchors_path = config.get("anchors_path")
        return SSDMobileNetV2Detector(
            model_path,
            labels,
            confidence_threshold=threshold,
            anchors_path=(
                resolve_config_path(anchors_path, config_path)
                if anchors_path else None
            ),
        )

    if backend == "yolov8_onnx":
        from noorix.yolov8_onnx import YOLOv8ONNXDetector

        labels = config.get("labels")
        labels_path = config.get("labels_path")
        if labels_path:
            resolved_labels = resolve_config_path(labels_path, config_path)
            labels = load_json_document(resolved_labels)
        if not isinstance(labels, list) or len(labels) != 80 or not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError(
                "YOLOv8 ONNX requires labels_path or exactly 80 COCO class labels"
            )
        return YOLOv8ONNXDetector(
            model_path,
            labels,
            confidence_threshold=threshold,
            iou_threshold=float(config.get("iou_threshold", 0.45)),
            input_size=int(config.get("input_size", 640)),
            max_detections=int(config.get("max_detections", 100)),
        )

    if backend == "opencv_dnn":
        labels = config.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError("OpenCV DNN requires a non-empty labels list")
        return ObjectDetector(
            model_path,
            labels,
            confidence_threshold=threshold,
            input_size=tuple(config.get("input_size", (300, 300))),
        )

    raise ValueError(f"Unsupported object detection backend: {backend}")
