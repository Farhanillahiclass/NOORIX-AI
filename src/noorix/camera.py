"""Explicitly managed webcam capture with deterministic cleanup."""
from __future__ import annotations

from typing import Any, Callable


class WebcamCapture:
    def __init__(self, device_index: int = 0, capture_factory: Callable[[int], Any] | None = None):
        if device_index < 0:
            raise ValueError("device_index must be zero or greater")
        self.device_index = device_index
        self._capture_factory = capture_factory
        self._capture = None

    @property
    def is_running(self) -> bool:
        return self._capture is not None

    def start(self) -> None:
        if self._capture is not None:
            return
        factory = self._capture_factory
        if factory is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "Camera support requires: python -m pip install -e '.[camera]'"
                ) from exc
            factory = cv2.VideoCapture
        capture = factory(self.device_index)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise RuntimeError(f"Could not open webcam device {self.device_index}")
        self._capture = capture

    def read(self):
        if self._capture is None:
            raise RuntimeError("Camera is not started; call start() first")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the webcam")
        return frame

    def stop(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
