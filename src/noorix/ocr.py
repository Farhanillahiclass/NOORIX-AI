"""Local printed-English OCR adapter backed by the Tesseract executable.

Install the optional Python extra and the Tesseract OCR application separately.
Only clear printed text is in scope; handwriting and Urdu OCR are not claimed.
"""
from __future__ import annotations


class PrintedEnglishOCR:
    def __init__(self, language: str = "eng", config: str = ""):
        if language != "eng":
            raise ValueError("This prototype currently supports English OCR only")
        self.language = language
        self.config = config

    def read(self, image) -> str:
        if image is None:
            raise ValueError("image must not be None")
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("OCR requires: pip install -e '.[ocr]'") from exc
        try:
            return pytesseract.image_to_string(image, lang=self.language, config=self.config).strip()
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract executable was not found. Install Tesseract OCR and add it to PATH."
            ) from exc
