# NOORIX AI
**See Beyond. Understand More.**  
**Powered by MRS**

NOORIX AI is an offline-first research prototype for camera-based accessibility assistance. Its intended modules are Pakistani banknote denomination estimation, everyday object detection with image-relative left/centre/right labels, printed English OCR and spoken feedback.

## One-command desktop start (Windows Git Bash)

After cloning the repository and activating the Python environment that contains your local NOORIX model dependencies, run this single command from Git Bash:

```bash
bash NOORIX.sh
```

The launcher selects the active `python`, sets `PYTHONPATH`, detects `D:/NOORIX_MODELS/EXTRACTED`, preserves an existing `config.json` (or creates one from the example), checks Python runtime imports, installs the declared extras only when required, checks Tkinter/Tesseract, and opens the desktop dashboard. The first dependency installation may require internet; after installation, inference uses local assets. The app does not download model weights at startup.

Inside the dashboard, open **Testing Center** and choose **RUN ALL SMOKE TESTS**. Then validate each capability with real camera frames, printed pages and representative audio. A green smoke-test result is not a production-readiness certificate. Handwriting OCR remains unavailable until a compatible trained model and adapter are supplied; Urdu OCR requires Tesseract's `urd` language data. The PKR classifier estimates denomination/side and does not authenticate notes.

## Run locally
Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev,camera,speech,ocr,onnx,asr-whisper]"
python -m pytest -q
```

Run the desktop interface:

```bash
python -m noorix --gui
```

The desktop dashboard includes an Overview workspace, live camera preview, object detection, configurable printed OCR language (English / Urdu / both when Tesseract language data is installed), PKR currency workspace, local Whisper audio-file transcription, system-voice TTS, Agent Console, Model Management, Model Testing Center, System Diagnostics, Activity Logs, Settings and About the creator. Open **Model Testing** in the sidebar to run all six configured smoke tests or run Package, PKR, YOLO, Printed OCR, Whisper ASR and TTS individually. JSON and Markdown reports are created under `reports/model-tests/` and can be opened from the dashboard. Handwritten OCR and experimental Urdu OCR remain explicitly unverified and are not represented as ready capabilities. The dashboard's Model Management view distinguishes a model file being present from a model being validated by real inputs.

Run the camera preview only:

```bash
python -m noorix --camera-preview
```

In the preview window, press **Q** or **Esc**, or close the window. If the window does not have keyboard focus, click it first. **Ctrl+C** in the terminal is also handled as a safe exit. The camera is released in all exit paths.

Check installed components:

```bash
python -m noorix --check
```

Run all configured local model smoke tests in one sequence:

```bash
python scripts/test_all_models.py --config config.json
```

On Windows, double-click `RUN_ALL_TESTS.bat`. It automatically checks the common extracted package location `D:\\NOORIX_MODELS\\EXTRACTED`; otherwise set the `NOORIX_MODELS_DIR` environment variable to your extracted model package folder. Config-relative model paths take precedence. Results are saved under `reports/model-tests/` as JSON and Markdown. Tests run one at a time and show PASS, FAIL, SKIPPED or WARN. A skipped test means a file, runtime or language pack is unavailable; it is not counted as a pass.

The test suite performs basic local loading/inference checks for PKR, YOLO, OCR, Whisper and installed TTS voices, plus optional package checksum verification. It uses synthetic inputs for vision/audio smoke tests and does not evaluate model accuracy or play TTS audio. It is a developer diagnostic, not a production-readiness certificate.

Transcribe a local audio file with the configured Whisper model:

```bash
python -m noorix --transcribe "recording.wav" --config "config.json"
```

This command reads the file locally and prints the transcript and verification warning.

## Local specialist agents

The desktop app includes an Agent Console with bounded OCR, object detection, currency, speech, safety, system-health, frame-quality and voice-command parsing agents. A deterministic coordinator runs only the selected task and returns structured results. Read [docs/AGENTS.md](docs/AGENTS.md) for contracts and limitations. Recognition agents still require compatible trained ONNX model files; the agent framework does not create model accuracy by itself. The voice-command parser accepts already-transcribed text and returns a confirmation-gated intent. Optional local Whisper ASR can transcribe Urdu or English audio when a complete model directory is configured; it runs on CPU and does not download weights at inference time.

## Model configuration

For the packaged YOLOv8n model, set `object_detection.backend` to `yolov8_onnx` and provide its ONNX file plus the matching `labels.json`. The adapter supports the standard Ultralytics YOLOv8 Detect output only; see [the model integration contract](docs/MODEL_INTEGRATION.md#yolov8-onnx-object-detection). Confirm the exact ONNX export against Ultralytics before relying on its predictions.

Copy `config.example.json` to `config.json`. Supply compatible, licensed ONNX models and the exact class-label order from each model's training process. Relative model paths are resolved from the folder containing `config.json`, regardless of the folder from which NOORIX is launched. Do not guess labels or use arbitrary ONNX models: model input/output conventions differ. Currency scores are estimates, not note authentication.

The desktop camera workspace includes an optional Auto Scan mode. When enabled, it periodically runs frame quality, printed-page OCR, object detection and currency estimation against the current in-memory frame. Missing models are reported as unavailable; Auto Scan does not create recognition capability by itself. Stop Auto Scan or close the app to end the loop.

Set `ocr.language` in `config.json` to `eng`, `urd`, or `eng+urd`. Urdu OCR requires the matching Tesseract Urdu trained-data file to be installed; the default remains English. This extracts visible printed text from a book or page but does not identify a book's title, author or edition.

## Offline Whisper speech-to-text

Set `asr.model_path` to the extracted local Whisper model directory and `asr.processor_path` to its separate processor directory, and set `asr.language` to `urdu` or `english`. Install the optional `asr-whisper` extra. The adapter accepts a local audio filename or a waveform plus its sample rate, converts audio to mono 16 kHz, and loads the model with `local_files_only=True` on CPU. It does not capture microphone audio itself; a GUI or microphone recorder must supply the audio payload. Whisper is speech-to-text, not text-to-speech. The existing `SpeechAgent`/pyttsx3 adapter remains a separate output path. Model-level Urdu accuracy and CPU latency must be measured on representative recordings before relying on transcripts.

## External requirements
- OCR: install the Tesseract application separately and ensure `tesseract --version` works in the terminal.
- Speech: pyttsx3 uses voices installed on the computer. Auto Scan looks for an installed Urdu-compatible voice before speaking Urdu summaries; if none is found, it reports that limitation.
- Recognition: no trained currency or object model weights are bundled. The adapters require compatible model files.
- GUI: Tkinter is included with most standard Python Windows installations.

## Privacy and safety

Before changing repository visibility to Public, review the [Public Release Security Checklist](docs/PUBLIC_RELEASE_CHECKLIST.md).
- Camera frames are held in memory for the current session and are not saved automatically.
- Left/centre/right describes only position within the camera image. It is not physical distance, obstacle avoidance or safe-navigation guidance.
- NOORIX is not a replacement for a white cane, guide dog or trained mobility support.
- OCR is for visible printed text; handwriting and book metadata recognition are not claimed. Urdu OCR requires installed Tesseract Urdu language data.
- Do not rely on this prototype for safety-critical decisions. Test results in supervised, controlled conditions.

## Project status

Camera lifecycle, preview, desktop shell, local specialist-agent framework, backend adapters, configuration template and automated tests are implemented. Trained model assets, real-world accuracy evaluation, Urdu voice verification and fully offline packaging remain pending. This is not yet a production-ready accessibility device.

## License

No license has been selected. Standard copyright applies; reuse is not automatically permitted.
