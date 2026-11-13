# NOORIX AI
**See Beyond. Understand More.**  
**Powered by MRS**

NOORIX AI is an offline-first research prototype for camera-based accessibility assistance. It is designed to explore Pakistani banknote denomination estimation, everyday object detection with image-relative left/centre/right labels, printed English OCR, and spoken feedback.

> **Current status:** webcam adapter and preview have been exercised locally by the project developer; backend adapters are implemented, but trained recognition models, model validation, and full offline packaging are still pending.

## Current modules
- Webcam capture adapter using OpenCV. The preview can be launched with `python -m noorix --camera-preview` in a local checkout that has the preview-enabled application entry point.
- Printed English OCR adapter using Tesseract. Install both the Python extra and the Tesseract application.
- Offline speech adapter using locally installed system voices through pyttsx3. Urdu support depends on a suitable voice installed on the computer and must be verified.
- Pakistani currency classifier adapter for a supplied ONNX classification model.
- Everyday object detector adapter for a supplied SSD-style ONNX model.
- Image-relative left/centre/right position helper.

## Install
Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -e ".[dev,camera,speech,ocr,onnx]"
python -m pytest -q
```

The `all` extra can also be installed with `python -m pip install -e ".[all,dev]"`.

### External requirements
- OCR: install the Tesseract OCR application separately and ensure its executable is on PATH.
- Object detection and currency estimation: supply compatible, trained ONNX model files and the exact class-label order used during training. No model weights are bundled.
- Speech: pyttsx3 uses installed system voices. Urdu speech is not guaranteed by installing the Python package alone.

## Safety and limitations
- Left/centre/right describes only an object's position in the camera image. It is not physical distance, obstacle avoidance, or safe-navigation guidance.
- NOORIX AI is not a replacement for a white cane, guide dog, or trained mobility support.
- Currency output is an estimate, not banknote authentication or counterfeit detection.
- OCR currently targets clear printed English; handwriting and Urdu OCR are not claimed.
- Model outputs must be validated on representative real-world data before any public demonstration. Do not rely on this prototype for safety-critical decisions.

## Project layout
- `src/noorix/`: application and modality adapters.
- `tests/`: automated tests.
- `docs/`: scope, architecture, safety and development notes.

Run tests with `python -m pytest -q`.

## License
No license has been selected yet. Until a license is added, standard copyright applies and reuse is not automatically permitted.
