"""Strict YOLOv8 detection-head decoder for Ultralytics ONNX exports.

Supports the standard, non-segment, non-pose YOLOv8 Detect output:
[1, 4 + class_count, candidate_count] or its transposed equivalent.
It deliberately rejects other model families/output contracts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from noorix.position import image_relative_position


class YOLOv8ONNXDetector:
    INPUT_SIZE = 640
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str],
        confidence_threshold: float = 0.35,
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.confidence_threshold = float(confidence_threshold)
        self._net = None
        if not self.labels or any(not isinstance(x, str) or not x.strip() for x in self.labels):
            raise ValueError("YOLOv8 labels must be a non-empty list of class names")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

    @staticmethod
    def read_labels(path: str | Path) -> list[str]:
        label_path = Path(path)
        if not label_path.is_file():
            raise FileNotFoundError(f"YOLOv8 labels file not found: {label_path}")
        data = json.loads(label_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            try:
                keys = sorted(data, key=lambda value: int(value))
                if [int(key) for key in keys] != list(range(len(keys))):
                    raise ValueError("YOLOv8 label indices must be contiguous from zero")
                labels = [data[key] for key in keys]
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLOv8 labels must use contiguous numeric class indices") from exc
        else:
            raise ValueError("YOLOv8 labels JSON must be a list or numeric-keyed object")
        if not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("YOLOv8 labels must contain non-empty class names")
        return labels

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLOv8 ONNX model not found: {self.model_path}")
        import cv2
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    @staticmethod
    def _letterbox(image):
        import cv2
        height, width = image.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 640 - resized_w, 640 - resized_h
        left, top = pad_w // 2, pad_h // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = resized
        return canvas, scale, left, top

    def _decode(self, output, scale: float, pad_x: int, pad_y: int, width: int, height: int):
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"YOLOv8 output must be rank 2 after batch removal; got {array.shape}")
        expected_features = 4 + len(self.labels)
        if array.shape[0] == expected_features:
            predictions = array.T
        elif array.shape[1] == expected_features:
            predictions = array
        else:
            raise ValueError(
                "Unsupported YOLOv8 output shape: expected one axis to equal "
                f"4 + {len(self.labels)} classes ({expected_features}), got {array.shape}. "
                "This decoder supports standard YOLOv8 Detect exports only."
            )
        if predictions.shape[0] == 0:
            return []
        if not np.isfinite(predictions).all():
            predictions = predictions[np.isfinite(predictions).all(axis=1)]
        if not len(predictions):
            return []

        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(predictions)), class_ids]
        candidate_ids = np.flatnonzero(scores >= self.confidence_threshold)
        boxes, confidences, classes = [], [], []
        for index in candidate_ids:
            cx, cy, box_w, box_h = (float(v) for v in predictions[index, :4])
            if box_w <= 0 or box_h <= 0:
                continue
            x1 = max(0.0, min(float(width), (cx - box_w / 2 - pad_x) / scale))
            y1 = max(0.0, min(float(height), (cy - box_h / 2 - pad_y) / scale))
            x2 = max(0.0, min(float(width), (cx + box_w / 2 - pad_x) / scale))
            y2 = max(0.0, min(float(height), (cy + box_h / 2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(scores[index]))
            classes.append(int(class_ids[index]))

        if not boxes:
            return []
        import cv2
        selected = []
        for class_id in sorted(set(classes)):
            local = [i for i, value in enumerate(classes) if value == class_id]
            indices = cv2.dnn.NMSBoxes(
                [boxes[i] for i in local],
                [confidences[i] for i in local],
                self.confidence_threshold,
                self.IOU_THRESHOLD,
            )
            if len(indices):
                selected.extend(local[int(i)] for i in np.asarray(indices).reshape(-1))
        selected.sort(key=lambda i: confidences[i], reverse=True)
        results = []
        for i in selected[:self.MAX_DETECTIONS]:
            x, y, box_w, box_h = boxes[i]
            x1, y1, x2, y2 = round(x), round(y), round(x + box_w), round(y + box_h)
            centre_x = (x1 + x2) / 2
            results.append({
                "label": self.labels[classes[i]],
                "class_id": classes[i],
                "confidence": confidences[i],
                "position": image_relative_position(centre_x, width),
                "box": [x1, y1, x2, y2],
            })
        return results

    def detect(self, image) -> list[dict[str, object]]:
        if image is None:
            raise ValueError("image must not be None")
        if not hasattr(image, "shape") or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("image must be a three-channel BGR image")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be greater than zero")
        if self._net is None:
            self.load()
        import cv2
        padded, scale, pad_x, pad_y = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1.0 / 255.0,
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(names)
        if isinstance(outputs, (list, tuple)):
            if len(outputs) != 1:
                raise ValueError(
                    "YOLOv8 Detect export must expose exactly one output tensor; "
                    f"received {len(outputs)}"
                )
            outputs = outputs[0]
        return self._decode(outputs, scale, pad_x, pad_y, width, height)


def create_yolov8_detector(config: dict, config_path: str | Path = "config.json"):
    from noorix.paths import resolve_config_path
    model_value = config.get("model_path")
    labels_value = config.get("labels_path")
    if not model_value or not labels_value:
        raise ValueError("YOLOv8 requires model_path and labels_path")
    model_path = resolve_config_path(model_value, config_path)
    labels_path = resolve_config_path(labels_value, config_path)
    labels = YOLOv8ONNXDetector.read_labels(labels_path)
    return YOLOv8ONNXDetector(
        model_path,
        labels,
        confidence_threshold=float(config.get("confidence_threshold", 0.35)),
    )
