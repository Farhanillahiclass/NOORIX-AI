"""OpenCV DNN adapter for SSD-style ONNX object detectors.

Model weights and labels are not bundled. The adapter expects an SSD detector
whose output follows [image_id, class_id, confidence, x1, y1, x2, y2].
"""
from __future__ import annotations

from pathlib import Path

from noorix.position import image_relative_position


class ObjectDetector:
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

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Object detection model not found: {self.model_path}")
        import cv2
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def detect(self, image) -> list[dict[str, object]]:
        if image is None:
            raise ValueError("image must not be None")
        if self._net is None:
            self.load()
        import cv2

        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 127.5, size=self.input_size,
            mean=(127.5, 127.5, 127.5), swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        detections = self._net.forward().reshape(-1, 7)
        results = []
        for row in detections:
            class_index = int(row[1])
            confidence = float(row[2])
            if confidence < self.confidence_threshold or not 0 <= class_index < len(self.labels):
                continue
            x1, y1, x2, y2 = (float(v) for v in row[3:7])
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
