"""Explicitly managed webcam capture with quality-oriented defaults and safe cleanup."""
from __future__ import annotations

from typing import Any, Callable


class WebcamCapture:
    """Manage a webcam and request a useful preview resolution when supported.

    Camera drivers may ignore requested properties. capture_info reports the
    actual dimensions so callers never assume that a request was accepted.
    """

    def __init__(self, device_index: int = 0, capture_factory: Callable[[int], Any] | None = None,
                 width: int = 1280, height: int = 720, autofocus: bool = True):
        if device_index < 0:
            raise ValueError("device_index must be zero or greater")
        if width < 160 or height < 120:
            raise ValueError("Requested camera dimensions are too small")
        self.device_index = device_index
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.autofocus = bool(autofocus)
        self._capture_factory = capture_factory
        self._capture = None
        self._cv2 = None
        self._release_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._capture is not None

    @property
    def capture_info(self) -> dict[str, int | bool | str]:
        """Return requested and actual capture properties where available."""
        info: dict[str, int | bool] = {
            "device_index": self.device_index,
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "autofocus_requested": self.autofocus,
        }
        if self._release_error is not None:
            info["release_error"] = self._release_error
        if self._capture is not None and self._cv2 is not None:
            try:
                info["actual_width"] = int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
                info["actual_height"] = int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
                if hasattr(self._cv2, "CAP_PROP_AUTOFOCUS"):
                    info["autofocus_enabled"] = bool(
                        self._capture.get(self._cv2.CAP_PROP_AUTOFOCUS)
                    )
            except Exception:
                pass
        return info

    def start(self) -> None:
        if self._capture is not None:
            return
        factory = self._capture_factory
        if factory is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Camera support requires: python -m pip install -e '.[camera]'") from exc
            self._cv2 = cv2
            factory = cv2.VideoCapture
        else:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                self._cv2 = None

        capture = factory(self.device_index)
        try:
            opened = capture is not None and capture.isOpened()
        except Exception as exc:
            if capture is not None:
                self._release_capture(capture)
            raise RuntimeError(
                f"Could not initialize webcam device {self.device_index}"
            ) from exc
        if not opened:
            if capture is not None:
                self._release_capture(capture)
            raise RuntimeError(f"Could not open webcam device {self.device_index}")
        self._release_error = None

        try:
            if self._cv2 is not None:
                capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                if self.autofocus and hasattr(self._cv2, "CAP_PROP_AUTOFOCUS"):
                    capture.set(self._cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            # Some webcam drivers do not expose these controls.
            pass
        self._capture = capture

    def read(self):
        if self._capture is None:
            raise RuntimeError("Camera is not started; call start() first")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the webcam")
        return frame

    def _release_capture(self, capture: Any) -> None:
        try:
            capture.release()
        except Exception as exc:
            # Detach the handle even if a driver reports a release failure.
            # Keep only the exception type; driver messages may contain local details.
            self._release_error = type(exc).__name__

    def stop(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            self._release_capture(capture)

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
