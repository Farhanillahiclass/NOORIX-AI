# NOORIX AI
**See Beyond. Understand More.**  
**Powered by MRS**

NOORIX AI is a research prototype for an offline, camera-based accessibility assistant. Its planned modules are Pakistani currency denomination estimation, everyday object detection with image-relative left/centre/right labels, printed English OCR, and spoken feedback.

> **Current status: scaffold initialized — prototype implementation and live hardware testing are pending.** This repository does not yet contain trained models or validated recognition performance.

## Prototype scope
- Camera input from a laptop webcam.
- Currency denomination estimation (not banknote authentication).
- Everyday object detection and approximate image-relative position.
- Printed English OCR as a bonus feature.
- Spoken feedback; Urdu output is planned for currency/object results.

## Safety and limitations
- Left/centre/right describes position in the camera frame only. It is not distance measurement, obstacle avoidance, or safe-navigation guidance.
- NOORIX AI is not a replacement for a white cane, guide dog, or trained mobility support.
- Currency output is an estimate and must not be treated as counterfeit detection or authentication.
- OCR scope is printed English only in this prototype plan; Urdu OCR is future work.
- Do not rely on this prototype for safety-critical decisions.

## Quick start
Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m noorix
```

The initial application reports module readiness and deliberately does not claim that recognition models are installed.

## Project layout
- `src/noorix/`: application and capability interfaces.
- `tests/`: automated tests.
- `docs/`: scope, architecture, safety and development notes.

## Development
Run tests with `python -m pytest`.

See [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/SAFETY.md](docs/SAFETY.md).

## License
No license has been selected yet. Until a license is added, standard copyright applies and reuse is not automatically permitted.
