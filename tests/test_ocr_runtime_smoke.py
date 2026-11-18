"""Optional end-to-end smoke test for the local Tesseract OCR runtime.

This test exercises a generated, high-contrast English text image through the
real OCR adapter, OCR specialist and vision coordinator. It skips cleanly when
Tesseract or its English language data is not installed.
"""
from __future__ import annotations

import shutil

import pytest


def _local_english_ocr_available():
    if not shutil.which("tesseract"):
        return False
    try:
        import pytesseract
        return "eng" in pytesseract.get_languages(config="")
    except Exception:
        return False


@pytest.mark.skipif(
    not _local_english_ocr_available(),
    reason="Tesseract executable and English language data are required",
)
def test_real_english_ocr_flows_through_vision_pipeline():
    from PIL import Image, ImageDraw
    from noorix.agents import AgentCoordinator, OCRAgent, SafetyAgent
    from noorix.ocr import PrintedEnglishOCR

    # Render at a larger size to keep this fixture independent of system fonts.
    small = Image.new("RGB", (500, 100), "white")
    draw = ImageDraw.Draw(small)
    draw.text((12, 35), "NOORIX OCR TEST 2026", fill="black")
    frame = small.resize((2000, 400), Image.Resampling.LANCZOS)

    coordinator = AgentCoordinator([
        OCRAgent(PrintedEnglishOCR(language="eng")),
        SafetyAgent(),
    ])
    results = coordinator.run_vision(frame)

    assert [result.agent for result in results] == ["ocr", "safety"]
    ocr_result = results[0]
    assert ocr_result.successful, ocr_result.summary
    assert isinstance(ocr_result.data, str)
    recognized = " ".join(ocr_result.data.upper().split())
    assert all(token in recognized for token in ("NOORIX", "OCR", "TEST")), (
        f"Expected test phrase was not recognized. OCR returned: {ocr_result.data!r}"
    )
    assert results[1].data == {"reviewed": 1}
