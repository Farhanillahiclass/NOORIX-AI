import pytest

np = pytest.importorskip("numpy")

from noorix.camera import WebcamCapture
from noorix.ocr import PrintedEnglishOCR


class FakeCapture:
    def __init__(self, index):
        self.index = index
        self.properties = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, key, value):
        self.properties[key] = value
        return True

    def get(self, key):
        return self.properties.get(key, 0)

    def read(self):
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_requests_hd_resolution_and_releases():
    created = []

    def factory(index):
        capture = FakeCapture(index)
        created.append(capture)
        return capture

    camera = WebcamCapture(capture_factory=factory)
    camera.start()

    assert camera.is_running
    assert camera.capture_info["requested_width"] == 1280
    assert camera.capture_info["requested_height"] == 720
    assert camera.read().shape == (720, 1280, 3)

    camera.stop()
    assert not camera.is_running
    assert created[0].released


def test_camera_rejects_invalid_requested_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        WebcamCapture(width=80, height=60)


def test_ocr_converts_opencv_bgr_to_rgb():
    pytest.importorskip("PIL")
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :] = (0, 0, 255)

    converted = PrintedEnglishOCR._to_pil(bgr)

    assert converted.getpixel((0, 0)) == (255, 0, 0)


def test_ocr_tries_enhanced_variants_and_returns_best_text(monkeypatch):
    pytest.importorskip("PIL")
    pytesseract = pytest.importorskip("pytesseract")

    calls = []
    monkeypatch.setattr(pytesseract, "get_languages", lambda config: ["eng", "urd"])

    def fake_image_to_string(image, lang, config):
        calls.append((image, lang, config))
        return ["Hi", "This is a printed page", "Page text"][len(calls) - 1]

    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)
    frame = np.full((80, 120, 3), 255, dtype=np.uint8)

    result = PrintedEnglishOCR(language="eng+urd").read(frame)

    assert result == "This is a printed page"
    assert len(calls) == 3
    assert all(call[1] == "eng+urd" for call in calls)
    assert "--oem 3" in calls[0][2]


def test_ocr_rejects_empty_or_unsupported_arrays():
    with pytest.raises(ValueError, match="non-empty"):
        PrintedEnglishOCR._to_pil(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError, match="channels"):
        PrintedEnglishOCR._to_pil(np.zeros((5, 5, 2), dtype=np.uint8))


def test_ocr_corrects_sideways_page_using_tesseract_osd(monkeypatch):
    pil = pytest.importorskip("PIL.Image")
    pytesseract = pytest.importorskip("pytesseract")
    calls = []
    monkeypatch.setattr(pytesseract, "get_languages", lambda config: ["eng", "urd"])

    monkeypatch.setattr(
        pytesseract,
        "image_to_osd",
        lambda image, config: "Page number: 0\nRotate: 90\nOrientation in degrees: 270",
    )

    def fake_image_to_string(image, lang, config):
        calls.append(image.size)
        return "Printed page text"

    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)
    source = pil.new("RGB", (120, 80), "white")

    result = PrintedEnglishOCR(language="eng").read(source)

    assert result == "Printed page text"
    assert calls
    assert calls[0] == (80, 120)


def test_ocr_orientation_detection_failure_falls_back(monkeypatch):
    pil = pytest.importorskip("PIL.Image")
    pytesseract = pytest.importorskip("pytesseract")
    monkeypatch.setattr(
        pytesseract,
        "image_to_osd",
        lambda image, config: (_ for _ in ()).throw(RuntimeError("OSD unavailable")),
    )
    source = pil.new("RGB", (120, 80), "white")

    corrected = PrintedEnglishOCR._correct_orientation(source, pytesseract)

    assert corrected.size == (120, 80)



def test_ocr_reports_missing_requested_language_data(monkeypatch):
    pytesseract = pytest.importorskip("pytesseract")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config: ["eng"])
    frame = np.full((40, 80, 3), 255, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Missing Tesseract language data: urd"):
        PrintedEnglishOCR(language="urd").read(frame)
