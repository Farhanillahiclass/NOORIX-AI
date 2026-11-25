"""Pakistani banknote denomination classifier interface.

A real ONNX model and its verified class labels must be supplied by the user.
Predictions are estimates only, never banknote authentication.
"""
from __future__ import annotations

from pathlib import Path


class CurrencyEstimator:
    def __init__(self, model_path: str | Path, labels: list[str], input_size: int = 224):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.input_size = input_size
        self._session = None
        if not self.labels or not all(isinstance(label, str) and label.strip() for label in self.labels):
            raise ValueError("labels must contain non-empty denomination strings")
        normalized_labels = [label.strip().casefold() for label in self.labels]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError("labels must be unique after whitespace and case normalization")
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")

    def load(self) -> None:
        self._session = None
        from noorix.paths import require_regular_file

        self.model_path = require_regular_file(self.model_path, label="Currency model")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX inference requires: pip install -e '.[onnx]'") from exc
        self._session = None
        session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Currency model must expose exactly one input tensor")
        input_shape = inputs[0].shape
        if (
            len(input_shape) != 4
            or input_shape[1] != 3
            or (isinstance(input_shape[0], int) and input_shape[0] != 1)
            or (isinstance(input_shape[2], int) and input_shape[2] != self.input_size)
            or (isinstance(input_shape[3], int) and input_shape[3] != self.input_size)
        ):
            raise ValueError(f"Expected currency input [1,3,{self.input_size},{self.input_size}], got {input_shape}")
        input_type = getattr(inputs[0], "type", None)
        if input_type is not None and input_type != "tensor(float)":
            raise ValueError(f"Currency model input must be float32, got {input_type}")

        outputs = session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("Currency model must expose exactly one output tensor")
        output_shape = outputs[0].shape
        if (
            len(output_shape) not in {1, 2}
            or (
                len(output_shape) == 2
                and isinstance(output_shape[0], int)
                and output_shape[0] != 1
            )
            or (
                isinstance(output_shape[-1], int)
                and output_shape[-1] != len(self.labels)
            )
        ):
            raise ValueError(f"Expected one currency score per label ({len(self.labels)}), got {output_shape}")
        output_type = getattr(outputs[0], "type", None)
        if output_type is not None and output_type != "tensor(float)":
            raise ValueError(f"Currency model output must be float32, got {output_type}")
        self._session = session

    def predict(self, image) -> dict[str, float | str]:
        if image is None:
            raise ValueError("image must not be None")
        if self._session is None:
            self.load()
        import cv2
        import numpy as np

        input_metadata = self._session.get_inputs()
        if len(input_metadata) != 1:
            raise RuntimeError("Currency model must expose exactly one input tensor")
        input_meta = input_metadata[0]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        outputs = self._session.run(None, {input_meta.name: blob})
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("Currency model must return exactly one score tensor")
        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if scores.size != len(self.labels):
            raise RuntimeError(f"Model returned {scores.size} scores for {len(self.labels)} labels")
        if not np.isfinite(scores).all():
            raise RuntimeError("Currency model returned non-finite scores")
        probabilities = np.exp(scores - scores.max())
        probabilities = probabilities / probabilities.sum()
        index = int(probabilities.argmax())
        return {"denomination": self.labels[index], "confidence": float(probabilities[index])}
