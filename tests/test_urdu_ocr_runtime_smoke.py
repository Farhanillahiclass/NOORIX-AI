"""Optional end-to-end smoke test for the local Urdu OCR runtime."""

from __future__ import annotations

import shutil

import pytest


EXPECTED_LINES = (
    "یہ ایک اردو ٹیسٹ ہے۔",
    "پاکستان ہمارا ملک ہے۔",
)


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
    from PIL import Image, ImageDraw, ImageFont, features

    if not features.check("raqm"):
        pytest.skip("Pillow RAQM support is required for Urdu rendering")

    import numpy as np

    from noorix.agents import AgentCoordinator, OCRAgent, SafetyAgent
    from noorix.ocr import PrintedEnglishOCR

    font_path = r"C:\Windows\Fonts\tahoma.ttf"
    try:
        font = ImageFont.truetype(font_path, 64)
    except OSError:
        pytest.skip("A suitable Urdu-capable Tahoma font is unavailable")

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
