"""CPU ONNX Runtime adapter for standard YOLOv8 detection exports.

Supported output: raw YOLOv8 detection tensor with 4 box values and 80 COCO
class scores, shaped [1, 84, N] or [1, N, 84]. This is not compatible with
segmentation, pose, end-to-end/NMS-included, or custom-class exports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from noorix.position import image_relative_position


class YOLOv8ONNXDetector:
    """Run a standard COCO-80 YOLOv8 ONNX detection model on CPU."""

    COCO_CLASS_COUNT = 80
    MAX_CANDIDATES = 10_000
    MAX_NMS_CANDIDATES = 3_000
    COCO_LABELS = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        max_detections: int = 100,
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
        ):
            raise ValueError("confidence_threshold must be a finite number between 0 and 1")
        if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
            raise ValueError("iou_threshold must be a finite number greater than 0 and at most 1")
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise ValueError("input_size must be an integer >= 32")
        if isinstance(max_detections, bool) or not isinstance(max_detections, int):
            raise ValueError("max_detections must be a positive integer")
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = input_size
        self.max_detections = max_detections
        self._session = None
        self._input_name = None

        if len(self.labels) != self.COCO_CLASS_COUNT:
            raise ValueError("YOLOv8 COCO detector requires exactly 80 class labels")
        if any(not isinstance(label, str) or not label.strip() for label in self.labels):
            raise ValueError("YOLOv8 COCO labels must be non-empty strings")
        if len(set(self.labels)) != self.COCO_CLASS_COUNT:
            raise ValueError("YOLOv8 COCO labels must be unique")
        if tuple(self.labels) != self.COCO_LABELS:
            raise ValueError(
                "YOLOv8 labels must match the canonical COCO-80 class order"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than 0 and at most 1")
        if self.input_size < 32 or self.max_detections < 1:
            raise ValueError("input_size must be >= 32 and max_detections must be positive")

    def load(self) -> None:
        self._session = None
        self._input_name = None
        from noorix.paths import require_regular_file

        self.model_path = require_regular_file(self.model_path, label="YOLOv8 ONNX model")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install ONNX support: pip install -e '.[onnx]'") from exc

        self._session = None
        self._input_name = None
        session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("YOLOv8 ONNX model must have exactly one input")
        shape = inputs[0].shape
        if (
            len(shape) != 4
            or shape[1] != 3
            or (isinstance(shape[0], int) and shape[0] != 1)
        ):
            raise ValueError(f"Expected NCHW model input [1,3,H,W], received {shape}")
        if isinstance(shape[2], int) and shape[2] != self.input_size:
            raise ValueError(f"Model height {shape[2]} does not match input_size {self.input_size}")
        if isinstance(shape[3], int) and shape[3] != self.input_size:
            raise ValueError(f"Model width {shape[3]} does not match input_size {self.input_size}")

        input_type = getattr(inputs[0], "type", None)
        if input_type is not None and input_type != "tensor(float)":
            raise ValueError(f"YOLOv8 input must be float32, got {input_type}")

        outputs = session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("YOLOv8 ONNX model must expose exactly one detection output")
        output_shape = outputs[0].shape
        if len(output_shape) != 3:
            raise ValueError(
                f"Expected one rank-3 YOLOv8 output [1,84,N] or [1,N,84], got {output_shape}"
            )
        if isinstance(output_shape[0], int) and output_shape[0] != 1:
            raise ValueError(f"YOLOv8 output batch must be 1, got {output_shape}")
        feature_axes = output_shape[1:]
        if 84 not in feature_axes:
            raise ValueError(
                "YOLOv8 output must contain an 84-value feature axis "
                f"(4 box + 80 classes), got {output_shape}"
            )
        candidate_axis = feature_axes[1] if feature_axes[0] == 84 else feature_axes[0]
        if isinstance(candidate_axis, int) and candidate_axis < 1:
            raise ValueError("YOLOv8 output must contain at least one candidate")
        if isinstance(candidate_axis, int) and candidate_axis > self.MAX_CANDIDATES:
            raise ValueError("YOLOv8 output exceeds the candidate safety limit")
        output_type = getattr(outputs[0], "type", None)
        if output_type is not None and output_type != "tensor(float)":
            raise ValueError(f"YOLOv8 output must be float32, got {output_type}")
        self._session = session
        self._input_name = inputs[0].name

    @staticmethod
    def _letterbox(image: np.ndarray, size: int):
        import cv2

        height, width = image.shape[:2]
        scale = min(size / width, size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height))
        pad_x = (size - resized_width) / 2
        pad_y = (size - resized_height) / 2
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        right, bottom = size - resized_width - left, size - resized_height - top
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        return canvas, scale, left, top

    @staticmethod
    def _normalize_output(output: np.ndarray) -> np.ndarray:
        predictions = np.asarray(output, dtype=np.float32)
        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported; received {predictions.shape}")
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f"Expected a 2D YOLOv8 output after batch removal, got {predictions.shape}")
        if predictions.shape[0] == 84:
            predictions = predictions.T
        elif predictions.shape[1] != 84:
            raise ValueError(
                "Expected raw YOLOv8 COCO output with 84 values per candidate "
                f"(4 box + 80 classes); got {predictions.shape}"
            )
        if predictions.shape[1] != 84:
            raise ValueError(f"Unexpected YOLOv8 prediction shape: {predictions.shape}")
        if predictions.shape[0] > YOLOv8ONNXDetector.MAX_CANDIDATES:
            raise ValueError("YOLOv8 output exceeds the candidate safety limit")
        return predictions

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        # Bound worst-case quadratic NMS work on CPU-only devices.
        order = np.argsort(-scores, kind="stable")[:YOLOv8ONNXDetector.MAX_NMS_CANDIDATES]
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_current = max(0, boxes[current, 2] - boxes[current, 0]) * max(
                0, boxes[current, 3] - boxes[current, 1]
            )
            area_rest = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = area_current + area_rest - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = rest[iou <= threshold]
        return keep

    def _postprocess(
        self, output: np.ndarray, original_shape: tuple[int, int],
        scale: float, pad_x: int, pad_y: int,
    ) -> list[dict[str, object]]:
        predictions = self._normalize_output(output)
        finite = np.isfinite(predictions).all(axis=1)
        predictions = predictions[finite]
        if not len(predictions):
            return []

        scores_all = predictions[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(scores_all)), class_ids]
        selected = scores >= self.confidence_threshold
        predictions, class_ids, scores = predictions[selected], class_ids[selected], scores[selected]
        if not len(predictions):
            return []

        xywh = predictions[:, :4]
        boxes = np.column_stack((
            xywh[:, 0] - xywh[:, 2] / 2,
            xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2,
            xywh[:, 1] + xywh[:, 3] / 2,
        ))
        height, width = original_shape
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        kept = []
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            class_keep = self._nms(boxes[indices], scores[indices], self.iou_threshold)
            kept.extend(indices[class_keep].tolist())
        kept.sort(key=lambda index: float(scores[index]), reverse=True)
        results = []
        for index in kept[:self.max_detections]:
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[index])
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            results.append({
                "label": self.labels[class_id],
                "class_id": class_id,
                "confidence": float(scores[index]),
                "position": image_relative_position((x1 + x2) / 2, width),
                "box": [x1, y1, x2, y2],
            })
        return results

    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy image array")
        if image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
            raise ValueError("image must be a non-empty three-channel BGR image")
        if self._session is None:
            self.load()

        canvas, scale, pad_x, pad_y = self._letterbox(image, self.input_size)
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        if len(outputs) != 1:
            raise ValueError(f"Expected one raw detection output, received {len(outputs)}")
        return self._postprocess(outputs[0], image.shape[:2], scale, pad_x, pad_y)
