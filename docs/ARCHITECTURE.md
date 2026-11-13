# Architecture

## Initial structure
The project uses a small Python package with separate capability adapters:
- `camera`: future webcam capture adapter.
- `currency`: future Pakistani banknote denomination estimator.
- `objects`: future everyday object detector.
- `position`: pure image-relative left/centre/right helper.
- `ocr`: future printed-English OCR adapter.
- `speech`: future speech output adapter.
- `app`: command-line readiness report.

## Data flow (planned)
Camera frame → selected model adapter(s) → normalized result → feedback formatter → speech adapter.

## Design constraints
- Recognition adapters must expose uncertainty and avoid silently guessing.
- Model files are kept separate from source code and are not committed until their licensing and size are reviewed.
- Network-dependent setup or model downloads must be clearly distinguished from offline runtime.
- The initial scaffold does not capture camera frames or perform recognition.
