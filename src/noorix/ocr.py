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
    def _document_crop(image):
        """Return a perspective-corrected page when a clear quadrilateral is found."""
        try:
            import cv2
            import numpy as np
            from PIL import Image

            rgb = np.asarray(image.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 45, 135)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            image_area = float(rgb.shape[0] * rgb.shape[1])
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
                if cv2.contourArea(contour) < image_area * 0.12:
                    continue
                perimeter = cv2.arcLength(contour, True)
                polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                points = polygon.reshape(4, 2).astype("float32")
                sums = points.sum(axis=1)
                differences = np.diff(points, axis=1).reshape(4)
                ordered = np.array([
                    points[np.argmin(sums)],       # top-left
                    points[np.argmin(differences)], # top-right
                    points[np.argmax(sums)],       # bottom-right
                    points[np.argmax(differences)] # bottom-left
                ], dtype="float32")
                tl, tr, br, bl = ordered
                width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                if width < 120 or height < 120:
                    continue
                target = np.array([
                    [0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]
                ], dtype="float32")
                transform = cv2.getPerspectiveTransform(ordered, target)
                warped = cv2.warpPerspective(rgb, transform, (width, height))
                return Image.fromarray(warped)
        except Exception:
            # Page detection is optional; fall back to full-frame OCR on any failure.
            return None
        return None

    @staticmethod
    def _prepare_variants(image):
        from PIL import Image, ImageEnhance, ImageOps

        document = PrintedEnglishOCR._document_crop(image)
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
        variants = [image]
        if document is not None:
            variants.append(document)
        variants.extend((enlarged, thresholded))
        return tuple(variants)

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
                psm = "6" if index < 3 else "11"
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
