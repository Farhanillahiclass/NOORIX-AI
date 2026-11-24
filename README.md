# NOORIX AI
**See Beyond. Understand More.**  
**Powered by MRS**

NOORIX AI is an offline-first research prototype for camera-based accessibility assistance. Its intended modules are Pakistani banknote denomination estimation, everyday object detection with image-relative left/centre/right labels, printed English OCR and spoken feedback.

## Run locally
Supported Python versions: 3.11, 3.12 and 3.13 (the versions covered by the current CI matrix).

```bash
# Create the environment with a supported interpreter (Python 3.12 shown):
# Windows:
py -3.12 -m venv .venv
# macOS/Linux:
# python3.12 -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
python -m pytest -q
```

For the final combined local validation, run:

```bash
python scripts/run_full_validation.py
```

This runs compilation, the full pytest suite, dependency checks, example-config parsing, notebook structure checks and an optional Tesseract inventory. It writes timestamped Markdown and JSON reports under reports/validation/, plus an offline dependency declaration audit under reports/dependencies/. See [Dependency Audit](docs/DEPENDENCY_AUDIT.md) and [One-command local validation](docs/LOCAL_VALIDATION.md) for an optional offline wheelhouse build, verification and installation procedure. Hosted GitHub Actions are paused to avoid consuming the account's remaining Actions allowance; do not manually dispatch the workflow unless the owner explicitly authorizes it.

Optional CPU object-detection timing report (after configuring a compatible model):

```bash
python scripts/benchmark_runtime.py --config config.json --image tests/assets/coco_sample.jpg --iterations 30 --warmup 5
```

The tool writes cold-start and warm-inference latency statistics under
`reports/benchmarks/`. It repeats one image and is not an accuracy,
memory-use, end-to-end, or accessibility-safety benchmark.

Run the desktop interface:

```bash
python -m noorix --gui
```

Run the camera preview only:

```bash
python -m noorix --camera-preview
```

In the preview window, press **Q** or **Esc**, or close the window. If the window does not have keyboard focus, click it first. **Ctrl+C** in the terminal is also handled as a safe exit. The camera is released in all exit paths.

Check installed components:

```bash
python -m noorix --check
```

## Local specialist agents

The desktop app includes an Agent Console with bounded OCR, object detection, currency, speech, safety, system-health, frame-quality and voice-command parsing agents. A deterministic coordinator runs only the selected task and returns structured results. Read [docs/AGENTS.md](docs/AGENTS.md) for contracts and limitations. Recognition agents still require compatible trained ONNX model files; the agent framework does not create model accuracy by itself. The voice-command parser accepts already-transcribed text and returns a confirmation-gated intent; microphone speech-to-text is a separate pending feature.

## Model configuration

Copy `config.example.json` to `config.json`. Supply compatible, licensed ONNX models and the exact class-label order from each model's training process. Relative model paths are resolved from the folder containing `config.json`, regardless of the folder from which NOORIX is launched. Do not guess labels or use arbitrary ONNX models: model input/output conventions differ. Currency scores are estimates, not note authentication.

The desktop camera workspace includes an optional Auto Scan mode. When enabled, it periodically runs frame quality, printed-page OCR, object detection and currency estimation against the current in-memory frame. Missing models are reported as unavailable; Auto Scan does not create recognition capability by itself. Stop Auto Scan or close the app to end the loop.

Set `ocr.language` in `config.json` to `eng`, `urd`, or `eng+urd`. Urdu OCR requires the matching Tesseract Urdu trained-data file to be installed; the default remains English. This extracts visible printed text from a book or page but does not identify a book's title, author or edition.

For the optional standard YOLOv8 COCO-80 ONNX backend, see [YOLOv8 ONNX setup and COCO validation](docs/OBJECT_DETECTION_MODEL.md). It uses CPU ONNX Runtime and requires a compatible local model file; the weights are not committed to Git. The existing SSD MobileNet V2 backend remains available.

## External requirements
- OCR: install the Tesseract application separately and ensure `tesseract --version` works in the terminal.
- Speech: pyttsx3 uses voices installed on the computer. Auto Scan looks for an installed Urdu-compatible voice before speaking Urdu summaries; if none is found, it reports that limitation.
- Recognition: trained model weights are not bundled. The optional YOLOv8 ONNX adapter is implemented for a specific standard COCO-80 output contract; it still requires the matching local model file. Currency weights remain separately configured.
- GUI: Tkinter is included with most standard Python Windows installations.

## Model archive imports

Downloaded model ZIP files are treated as untrusted input. The importer applies path, file-type, size, compression-ratio and overwrite checks before extracting approved artifacts. Importing a file does not validate or activate the model. Review [Model Import Security](docs/MODEL_IMPORT_SECURITY.md) before importing third-party artifacts.

## Privacy and safety

Before changing repository visibility to Public, review the [Public Release Security Checklist](docs/PUBLIC_RELEASE_CHECKLIST.md).
- Camera frames are held in memory for the current session and are not saved automatically.
- Left/centre/right describes only position within the camera image. It is not physical distance, obstacle avoidance or safe-navigation guidance.
- NOORIX is not a replacement for a white cane, guide dog or trained mobility support.
- OCR is for visible printed text; handwriting and book metadata recognition are not claimed. Urdu OCR requires installed Tesseract Urdu language data.
- Do not rely on this prototype for safety-critical decisions. Test results in supervised, controlled conditions.

## Production readiness

Track release blockers and mandatory evidence in [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md). Use the [Recognition and Device Evaluation Protocol](docs/RECOGNITION_EVALUATION_PROTOCOL.md) for held-out accuracy, OCR, target-device and supervised accessibility evidence. NOORIX remains a research prototype until every mandatory release gate is verified.

## Project status

Camera lifecycle, preview, desktop shell, local specialist-agent framework, backend adapters, configuration template and automated tests are implemented. A standard YOLOv8 COCO-80 ONNX inference adapter is available, but real-weight integration and real-dataset evaluation must be verified on the target machine. Real-world accuracy evaluation, Urdu voice verification and fully offline packaging remain pending. This is not yet a production-ready accessibility device.

## License

No license has been selected. Standard copyright applies; reuse is not automatically permitted.
