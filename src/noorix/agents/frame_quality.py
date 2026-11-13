"""Local camera-frame quality checks for more reliable recognition attempts."""
from __future__ import annotations

from typing import Any

from noorix.agents.base import AgentResult, AgentState


class FrameQualityAgent:
    """Assess basic image suitability; never infer scene or navigation safety."""

    name = "frame_quality"
    description = "Check image dimensions, brightness and blur before recognition."

    def __init__(
        self,
        min_width: int = 160,
        min_height: int = 120,
        min_brightness: float = 35.0,
        max_brightness: float = 225.0,
        min_sharpness: float = 50.0,
    ):
        if min_width < 1 or min_height < 1:
            raise ValueError("Minimum frame dimensions must be positive")
        if not 0 <= min_brightness < max_brightness <= 255:
            raise ValueError("Brightness thresholds must be ordered within 0..255")
        if min_sharpness < 0:
            raise ValueError("min_sharpness must not be negative")
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness

    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "A camera frame is required.")
        try:
            import cv2
            import numpy as np

            frame = np.asarray(payload)
            if frame.ndim not in (2, 3) or frame.size == 0:
                return AgentResult(self.name, AgentState.NEEDS_SETUP, "The supplied frame is empty or malformed.")
            height, width = frame.shape[:2]
            if frame.ndim == 3 and frame.shape[2] not in (3, 4):
                return AgentResult(self.name, AgentState.NEEDS_SETUP, "Expected a grayscale, BGR or BGRA image.")
            gray = frame if frame.ndim == 2 else cv2.cvtColor(
                frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            )
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            warnings = []
            if width < self.min_width or height < self.min_height:
                warnings.append("Frame resolution is low; recognition may be unreliable.")
            if brightness < self.min_brightness:
                warnings.append("Frame appears too dark; recognition may be unreliable.")
            elif brightness > self.max_brightness:
                warnings.append("Frame appears overexposed; recognition may be unreliable.")
            if sharpness < self.min_sharpness:
                warnings.append("Frame may be blurry; recognition may be unreliable.")
            summary = "Frame quality checks completed." if not warnings else "Frame quality needs attention."
            return AgentResult(
                self.name, AgentState.COMPLETED, summary,
                data={
                    "width": int(width), "height": int(height),
                    "brightness_mean": round(brightness, 2),
                    "sharpness_laplacian_variance": round(sharpness, 2),
                    "suitable_for_attempt": not warnings,
                },
                warnings=tuple(warnings),
            )
        except ImportError as exc:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, f"Frame quality requires OpenCV and NumPy: {exc}")
        except (TypeError, ValueError) as exc:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, f"Invalid camera frame: {exc}")
