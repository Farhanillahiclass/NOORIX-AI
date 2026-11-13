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
        if not self.labels:
            raise ValueError("labels must contain at least one denomination")
        if input_size <= 0:
            raise ValueError("input_size must be positive")

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Currency model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX inference requires: pip install -e '.[onnx]'") from exc
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def predict(self, image) -> dict[str, float | str]:
        if image is None:
            raise ValueError("image must not be None")
        if self._session is None:
            self.load()
        import cv2
        import numpy as np

        input_meta = self._session.get_inputs()[0]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        outputs = self._session.run(None, {input_meta.name: blob})
        scores = np.asarray(outputs[0]).reshape(-1)
        if scores.size != len(self.labels):
            raise RuntimeError(f"Model returned {scores.size} scores for {len(self.labels)} labels")
        probabilities = np.exp(scores - scores.max())
        probabilities = probabilities / probabilities.sum()
        index = int(probabilities.argmax())
        return {"denomination": self.labels[index], "confidence": float(probabilities[index])}
