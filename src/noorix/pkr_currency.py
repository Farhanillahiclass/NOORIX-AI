"""PyTorch MobileNetV2 adapter for Pakistani banknote denomination/side estimates.

This model does not authenticate banknotes or detect counterfeit notes.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

class PytorchPKRCurrencyEstimator:
    def __init__(self, model_path: str | Path, labels_path: str | Path, input_size: int = 224):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.input_size = int(input_size)
        self.labels: list[str] = []
        self._model: Any | None = None
        self._torch: Any | None = None
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")

    @staticmethod
    def _read_labels(path: Path) -> list[str]:
        if not path.is_file():
            raise FileNotFoundError(f"Currency labels not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = data.get("classes") if isinstance(data, dict) else None
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels.json must contain a non-empty 'classes' list")
        if not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError("Every currency class label must be a non-empty string")
        if len(set(labels)) != len(labels):
            raise ValueError("Currency class labels must be unique")
        return labels

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Currency model not found: {self.model_path}")
        labels = self._read_labels(self.labels_path)
        try:
            import torch
            from torchvision.models import mobilenet_v2
        except ImportError as exc:
            raise RuntimeError("PyTorch PKR inference requires torch and torchvision.") from exc
        model = mobilenet_v2(weights=None, num_classes=len(labels))
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("This PyTorch version must support safe weights_only loading.") from exc
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid PKR checkpoint: expected a state dictionary")
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        self.labels = labels
        self._torch = torch
        self._model = model

    @staticmethod
    def _describe_class(label: str) -> tuple[str, str]:
        match = re.fullmatch(r"(\d+)_(front|back)", label.strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"Unsupported PKR class label: {label}")
        return match.group(1), match.group(2).lower()

    def predict(self, image) -> dict[str, float | str]:
        if image is None:
            raise ValueError("image must not be None")
        if self._model is None:
            self.load()
        import cv2
        import numpy as np
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image must be a three-channel OpenCV image")
        if image.shape[2] != 3:
            raise ValueError("image must have exactly three color channels")
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        torch = self._torch
        input_tensor = torch.from_numpy(np.ascontiguousarray(tensor))
        with torch.inference_mode():
            logits = self._model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())
        label = self.labels[index]
        denomination, side = self._describe_class(label)
        return {"denomination": denomination, "side": side, "label": label, "confidence": confidence}
