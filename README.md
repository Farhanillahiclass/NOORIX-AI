# NOORIX AI
**See Beyond. Understand More.**  
**Powered by MRS**

NOORIX AI is an offline-first research prototype for camera-based accessibility assistance. Its intended modules are Pakistani banknote denomination estimation, everyday object detection with image-relative left/centre/right labels, printed English OCR and spoken feedback.

## Run locally
Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,camera,speech,ocr,onnx]"
python -m pytest -q
```

Run the desktop interface:

```bash
python -m noorix --gui
``

Run the camera preview only:

```bash
python -m noorix --camera-preview
``

In the preview window, press **Q** or **Esc**, or close the window. If the window does not have keyboard focus, click it first. **Ctrl+C** in the terminal is also handled as a safe exit. The camera is released in all exit paths.

Check installed components:

```bash
python -m noorix --check
```

## Model configuration
Copy `config.example.json` to `config.json`. Supply compatible, licensed ONNX models and the exact class-label order from each model's training process. Do not guess labels or use arbitrary ONNX models: model input/output conventions differ. Currency scores are estimates, not note authentication.

## External requirements
- OCR: install the Tesseract application separately and ensure `tesseract --version` works in the terminal.
- Speech: pyttsx3 uses voices installed on the computer. Urdu speech is not guaranteed; verify an Urdu-capable local voice.
- Recognition: no trained currency or object model weights are bundled. The adapters require compatible model files.
- GUI: Tkinter is included with most standard Python Windows installations.

## Privacy and safety
- Camera frames are held in memory for the current session and are not saved automatically.
- Left/centre/right describes only position within the camera image. It is not physical distance, obstacle avoidance or safe-navigation guidance.
- NOORIX AI is not a replacement for a white cane, guide dog or trained mobility support.
- OCR scope is clear printed English; handwriting and Urdu OCR are not claimed.
- Do not rely on this prototype for safety-critical decisions. Test results in supervised, controlled conditions.

## Project status
Camera lifecycle, preview, desktop shell, backend adapters, configuration template and automated tests are implemented. Trained model assets, real-world accuracy evaluation, Urdu voice verification and fully offline packaging remain pending. This is not yet a production-ready accessibility device.

## License
No license has been selected. Standard copyright applies; reuse is not automatically permitted.
