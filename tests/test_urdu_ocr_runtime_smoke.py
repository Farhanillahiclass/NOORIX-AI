"""Optional end-to-end smoke test for the local Urdu OCR runtime."""

from __future__ import annotations

import os
import shutil

import pytest


EXPECTED_LINES = (
    "یہ ایک اردو ٹیسٹ ہے۔",
    "پاکستان ہمارا ملک ہے۔",
)


def _load_urdu_font():
    """Find a local font with Arabic-script glyphs without downloading assets."""
    from PIL import ImageFont

    candidates = (
        os.environ.get("NOORIX_URDU_TEST_FONT"),
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, 64)
        except OSError:
            continue
    return None


def _local_urdu_ocr_available():
    if not shutil.which("tesseract"):
        return False

    try:
        import pytesseract

        return "urd" in pytesseract.get_languages(config="")
    except Exception:
        return False


@pytest.mark.skipif(
    not _local_urdu_ocr_available(),
    reason="Tesseract executable and Urdu language data are required",
)
def test_real_urdu_ocr_flows_through_vision_pipeline():
    from PIL import Image, ImageDraw, features

    if not features.check("raqm"):
        pytest.skip("Pillow RAQM support is required for Urdu rendering")

    import numpy as np

    from noorix.agents import AgentCoordinator, OCRAgent, SafetyAgent
    from noorix.ocr import PrintedEnglishOCR

    font = _load_urdu_font()
    if font is None:
        pytest.skip(
            "No supported local Urdu font found; set NOORIX_URDU_TEST_FONT to a font path"
        )

    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)

    for y, line in zip((100, 220), EXPECTED_LINES):
        draw.text(
            (1500, y),
            line,
            font=font,
            fill="black",
            direction="rtl",
            language="ur",
            anchor="ra",
        )

    coordinator = AgentCoordinator([
        OCRAgent(PrintedEnglishOCR(language="urd")),
        SafetyAgent(),
    ])

    frame = np.asarray(image)
    results = coordinator.run_vision(frame)

    assert [result.agent for result in results] == ["ocr", "safety"]

    ocr_result = results[0]
    assert ocr_result.successful, ocr_result.summary
    assert isinstance(ocr_result.data, str)

    recognized = " ".join(ocr_result.data.split())
    for expected in EXPECTED_LINES:
        assert expected in recognized, (
            f"Expected Urdu text was not recognized. "
            f"OCR returned: {ocr_result.data!r}"
        )

    assert results[1].successful, results[1].summary
    assert results[1].data == {"reviewed": 1}
