"""Image-relative position helpers; these are not navigation instructions."""

from typing import Literal

Position = Literal["left", "centre", "right"]


def image_relative_position(center_x: float, frame_width: float) -> Position:
    """Classify a normalized pixel x-coordinate into image thirds.

    This reports only where a detected object appears in the image.
    It does not estimate distance or whether a path is safe.
    """
    if frame_width <= 0:
        raise ValueError("frame_width must be greater than zero")
    if not 0 <= center_x <= frame_width:
        raise ValueError("center_x must be within the frame")
    fraction = center_x / frame_width
    if fraction < 1 / 3:
        return "left"
    if fraction > 2 / 3:
        return "right"
    return "centre"
