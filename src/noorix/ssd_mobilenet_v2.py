"""Verified post-processing for the SSD MobileNet V2 COCO ONNX model."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SSDMobileNetV2Detector:
    """Run the project's raw-head SSD MobileNet V2 ONNX model."""

    INPUT_SIZE = (300, 300)
    BOX_SCALES = (10.0, 10.0, 5.0, 5.0)
    SCORE_THRESHOLD = 0.30
    IOU_THRESHOLD = 0.60
    MAX_DETECTIONS_PER_CLASS = 100
    MAX_TOTAL_DETECTIONS = 100
    HEAD_SIZES = (19, 10, 5, 3, 2, 1)
    HEAD_ANCHORS = (3, 6, 6, 6, 6, 6)

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str | None],
        confidence_threshold: float = SCORE_THRESHOLD,
        anchors_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.confidence_threshold = float(confidence_threshold)
        self.anchors_path = (
            Path(anchors_path) if anchors_path is not None
            else self.model_path.with_name("ssd-mobilenet-v2-anchors.npz")
        )
        self._session = None
        self._anchors = None

        if len(self.labels) != 90:
            raise ValueError(
                "SSD MobileNet V2 COCO requires exactly 90 label positions in model order"
            )
        if any(
            label is not None and (not isinstance(label, str) or not label.strip())
            for label in self.labels
        ):
            raise ValueError("Each COCO label must be a non-empty string or None")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

    def load(self) -> None:
        self._session = None
        self._anchors = None
        from noorix.paths import require_regular_file

        self.model_path = require_regular_file(self.model_path, label="Object detection model")
        self.anchors_path = require_regular_file(self.anchors_path, label="SSD anchor file")

        import onnxruntime as ort

        with np.load(self.anchors_path) as data:
            anchors = np.asarray(data["anchors"], dtype=np.float32)

        if anchors.shape != (1917, 4) or not np.isfinite(anchors).all():
            raise ValueError("SSD anchor file must contain 1,917 finite [ymin,xmin,ymax,xmax] anchors")

        session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 300, 300]:
            raise ValueError("Unexpected SSD ONNX input; expected [1,3,300,300]")
        input_type = getattr(inputs[0], "type", None)
        if input_type is not None and input_type != "tensor(float)":
            raise ValueError(f"SSD ONNX input must be float32, got {input_type}")

        outputs = session.get_outputs()
        if len(outputs) != 12:
            raise ValueError(f"Expected 12 SSD prediction heads, received {len(outputs)}")
        expected_shapes = []
        for size, anchor_count in zip(self.HEAD_SIZES, self.HEAD_ANCHORS):
            expected_shapes.extend((
                [1, anchor_count * 4, size, size],
                [1, anchor_count * 91, size, size],
            ))
        for index, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            if output.shape != expected_shape:
                raise ValueError(f"Unexpected SSD output {index}; expected {expected_shape}, got {output.shape}")
            output_type = getattr(output, "type", None)
            if output_type is not None and output_type != "tensor(float)":
                raise ValueError(f"SSD output {index} must be float32, got {output_type}")

        self._session = session
        self._anchors = anchors

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        result = np.empty_like(values)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)
        return result

    def _decode_heads(self, outputs: list[np.ndarray]):
        if len(outputs) != 12:
            raise ValueError(f"Expected 12 output tensors, received {len(outputs)}")

        box_heads = []
        class_heads = []

        for index, (size, anchor_count) in enumerate(
            zip(self.HEAD_SIZES, self.HEAD_ANCHORS)
        ):
            boxes = np.asarray(outputs[index * 2], dtype=np.float32)
            classes = np.asarray(outputs[index * 2 + 1], dtype=np.float32)

            expected_box_shape = (1, anchor_count * 4, size, size)
            expected_class_shape = (1, anchor_count * 91, size, size)

            if boxes.shape != expected_box_shape:
                raise ValueError(
                    f"Invalid box head {index}: expected {expected_box_shape}, got {boxes.shape}"
                )
            if classes.shape != expected_class_shape:
                raise ValueError(
                    f"Invalid class head {index}: expected {expected_class_shape}, got {classes.shape}"
                )

            boxes = boxes.transpose(0, 2, 3, 1).reshape(1, -1, 4)
            classes = classes.transpose(0, 2, 3, 1).reshape(1, -1, anchor_count, 91)
            classes = classes.reshape(1, -1, 91)

            box_heads.append(boxes)
            class_heads.append(classes)

        encoded_boxes = np.concatenate(box_heads, axis=1)[0]
        logits = np.concatenate(class_heads, axis=1)[0]

        if encoded_boxes.shape != (1917, 4) or logits.shape != (1917, 91):
            raise ValueError("Decoded SSD heads do not match the expected 1,917-anchor layout")

        return encoded_boxes, logits

    def _decode_boxes(self, encoded_boxes: np.ndarray) -> np.ndarray:
        anchors = self._anchors
        ay = (anchors[:, 0] + anchors[:, 2]) / 2.0
        ax = (anchors[:, 1] + anchors[:, 3]) / 2.0
        ah = anchors[:, 2] - anchors[:, 0]
        aw = anchors[:, 3] - anchors[:, 1]

        ty, tx, th, tw = encoded_boxes.T
        cy = (ty / self.BOX_SCALES[0]) * ah + ay
        cx = (tx / self.BOX_SCALES[1]) * aw + ax
        height = np.exp(th / self.BOX_SCALES[2]) * ah
        width = np.exp(tw / self.BOX_SCALES[3]) * aw

        boxes = np.stack(
            (cy - height / 2, cx - width / 2,
             cy + height / 2, cx + width / 2),
            axis=1,
        )
        return np.clip(boxes, 0.0, 1.0)

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        top = np.maximum(box[0], boxes[:, 0])
        left = np.maximum(box[1], boxes[:, 1])
        bottom = np.minimum(box[2], boxes[:, 2])
        right = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0.0, bottom - top) * np.maximum(0.0, right - left)
        area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_boxes = (
            np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        )
        union = area_box + area_boxes - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        selected = []

        while order.size and len(selected) < self.MAX_DETECTIONS_PER_CLASS:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            overlaps = self._iou(boxes[current], boxes[remaining])
            order = remaining[overlaps <= self.IOU_THRESHOLD]

        return np.asarray(selected, dtype=np.int64)

    def _postprocess(self, encoded_boxes: np.ndarray, logits: np.ndarray):
        boxes = self._decode_boxes(encoded_boxes)
        foreground_scores = self._sigmoid(logits)[:, 1:]

        detections = []
        for class_index in range(90):
            scores = foreground_scores[:, class_index]
            candidate_indices = np.flatnonzero(scores > self.confidence_threshold)
            if candidate_indices.size == 0:
                continue

            selected_local = self._nms(boxes[candidate_indices], scores[candidate_indices])
            for local_index in selected_local:
                anchor_index = int(candidate_indices[local_index])
                detections.append((
                    float(scores[anchor_index]),
                    class_index,
                    boxes[anchor_index],
                ))

        detections.sort(key=lambda item: -item[0])
        return detections[:self.MAX_TOTAL_DETECTIONS]

    def detect(self, image) -> list[dict[str, object]]:
        if image is None:
            raise ValueError("image must not be None")
        if not hasattr(image, "shape") or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("image must be a three-channel BGR image")

        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be greater than zero")

        if self._session is None:
            self.load()

        import cv2

        resized = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 127.5 - 1.0
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})
        encoded_boxes, logits = self._decode_heads(outputs)
        detections = self._postprocess(encoded_boxes, logits)

        from noorix.position import image_relative_position

        results = []
        for confidence, class_index, box in detections:
            ymin, xmin, ymax, xmax = (float(value) for value in box)
            centre_x = (xmin + xmax) / 2.0 * width
            label = self.labels[class_index]
            results.append({
                "label": label or f"unknown_class_{class_index + 1}",
                "class_id": class_index + 1,
                "confidence": confidence,
                "position": image_relative_position(centre_x, width),
                "box": [
                    max(0, min(width, round(xmin * width))),
                    max(0, min(height, round(ymin * height))),
                    max(0, min(width, round(xmax * width))),
                    max(0, min(height, round(ymax * height))),
                ],
            })
        return results
