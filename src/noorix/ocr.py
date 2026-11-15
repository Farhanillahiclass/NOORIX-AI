"""Local printed-text OCR with conservative image enhancement.

Tesseract and its language data are installed separately. This adapter supports
printed English and Urdu; handwriting and guaranteed recognition are out of scope.
"""
from __future__ import annotations


class PrintedEnglishOCR:
    """Read printed text from PIL images or OpenCV BGR/BGRA frames."""

    def __init__(self, language: str = "eng", config: str = ""):
        supported = {"eng", "urd", "eng+urd"}
        if language not in supported:
            raise ValueError("language must be 'eng', 'urd', or 'eng+urd'")
        self.language = language
        self.config = config

    @staticmethod
    def _to_pil(image):
        from PIL import Image
        import numpy as np

        if isinstance(image, Image.Image):
            return image.copy()
        array = np.asarray(image)
        if array.size == 0 or array.ndim not in (2, 3):
            raise ValueError("image must be a non-empty grayscale, BGR, or BGRA image")
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.shape[2] == 3:
            return Image.fromarray(array[:, :, ::-1].copy()).convert("RGB")
        if array.shape[2] == 4:
            return Image.fromarray(array[:, :, [2, 1, 0, 3]].copy()).convert("RGB")
        raise ValueError("image must have 1, 3, or 4 channels")

    @staticmethod
    def _prepare_variants(image):
        from PIL import Image, ImageEnhance, ImageOps

        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        enhanced = ImageEnhance.Sharpness(gray).enhance(1.35)
        width, height = enhanced.size
        enlarged = enhanced.resize(
            (min(width * 2, 3200), min(height * 2, 3200)),
            Image.Resampling.LANCZOS,
        )
        try:
            import cv2
            import numpy as np
            values = np.asarray(enlarged)
            _, binary = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded = Image.fromarray(binary)
        except ImportError:
            thresholded = enlarged.point(lambda value: 255 if value > 160 else 0)
        return (image, enlarged, thresholded)

    @staticmethod
    def _text_score(text: str) -> tuple[int, int]:
        compact = "".join(character for character in text if not character.isspace())
        meaningful = sum(
            character.isalnum() or "\u0600" <= character <= "\u06ff"
            for character in compact
        )
        return meaningful, len(compact)

    def read(self, image) -> str:
        if image is None:
            raise ValueError("image must not be None")
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("OCR requires: pip install -e '.[ocr]'") from exc

        try:
            pil_image = self._to_pil(image)
            variants = self._prepare_variants(pil_image)
            base_config = self.config.strip()
            candidates = []
            for index, candidate in enumerate(variants):
                psm = "6" if index < 2 else "11"
                config = " ".join(part for part in ("--oem 3", f"--psm {psm}", base_config) if part)
                text = pytesseract.image_to_string(candidate, lang=self.language, config=config).strip()
                if text:
                    candidates.append(text)
            if not candidates:
                return ""
            return max(candidates, key=self._text_score)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract executable was not found. Install Tesseract OCR and add it to PATH."
            ) from exc
