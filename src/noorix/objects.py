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
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Object detection model not found: {self.model_path}")
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
            results.append({
                "label": self.labels[class_index],
                "confidence": confidence,
                "position": image_relative_position(centre_x, width),
                "box": [
                    max(0, min(width, int(x1 * width))),
                    max(0, min(height, int(y1 * height))),
                    max(0, min(width, int(x2 * width))),
                    max(0, min(height, int(y2 * height))),
                ],
            })
        return results
