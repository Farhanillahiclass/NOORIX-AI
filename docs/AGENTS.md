# NOORIX AI Agent Architecture

NOORIX uses a deterministic local specialist-agent design. "Agent" means a
bounded software component with a defined input/output contract; it does not
imply an autonomous LLM or unrestricted tool access.

## Specialist agents

| Agent | Input | Responsibility |
|---|---|---|
| OCR | Camera frame | Read visible printed English and/or Urdu using configured Tesseract language data |
| Object Detection | Camera frame | Run a configured compatible ONNX detector; report image-relative positions |
| Currency | Camera frame | Run configured ONNX classifier; estimate denomination only |
| Speech | Text | Speak through an installed local system voice |
| Safety | Specialist results | Attach conservative warnings and disclose failed tasks |
| System Health | Local configuration | Report adapter, executable and model-file readiness |
| Frame Quality | Camera frame | Check dimensions, brightness and blur; report image suitability only |
| Voice Command | Transcribed text | Map a small English/Roman Urdu/Urdu allowlist to an intent; require caller confirmation |

## Coordinator

AgentCoordinator registers agents by unique name and executes only explicitly
requested agents. Its vision pipeline runs the registered frame-quality, OCR,
object-detection and currency agents in that order, then passes all produced
results to the Safety Agent. Each specialist is optional: unregistered agents
are skipped, while setup and runtime failures are returned as structured
results rather than silently omitted. The voice-command parser accepts
already-transcribed text; microphone capture and speech-to-text are separate
capabilities. No agent downloads files, sends frames to a server, or saves
images.

## Honest readiness states

- ready: configured and available to attempt a task.
- running: reserved for UI progress indication.
- completed: task returned a result; this is not a claim of model accuracy.
- needs_setup: required frame, model, labels, executable or voice is missing.
- error: task failed and the UI should show the reason.

A model adapter is not production-ready until exact preprocessing, output shape,
label order and held-out evaluation are verified.

The optional real-runtime OCR smoke test is `tests/test_ocr_runtime_smoke.py`.
It creates a synthetic high-contrast English text image and sends it through the
real Tesseract adapter, OCR Agent and Safety Agent. It skips when Tesseract or
its English (`eng`) language data is unavailable. Run it with:

`python -m pytest tests/test_ocr_runtime_smoke.py -q`

This is a runtime integration check, not a general OCR accuracy benchmark.

## Safety boundary

Image-relative left/centre/right is not distance estimation, obstacle avoidance
or navigation. Currency output is not banknote authentication. OCR can misread
or omit text, especially on blurred, small, stylized or low-contrast content.
NOORIX must never be presented as a replacement for a white cane, guide dog or
trained mobility support.

## Urdu OCR runtime smoke test

NOORIX supports Urdu OCR through the `PrintedEnglishOCR` adapter when
Tesseract's Urdu (`urd`) language data is installed.

Run the optional end-to-end Urdu OCR runtime test with:

`python -m pytest tests/test_urdu_ocr_runtime_smoke.py -q`

The test generates a synthetic Urdu image and checks recognition through
`OCRAgent`, `AgentCoordinator.run_vision()`, and `SafetyAgent`.

Requirements:
- Tesseract executable available on PATH.
- Tesseract Urdu (`urd`) language data installed.
- Pillow with RAQM support.
- A local Urdu-capable font. The test checks common Tahoma/Arial, Noto Naskh,
  Noto Sans Arabic, DejaVu Sans and FreeSans paths. Set
  `NOORIX_URDU_TEST_FONT` to an explicit font file when it is installed
  elsewhere. No font is downloaded by the test.

The test skips when required local OCR or image-rendering dependencies are
unavailable. A passing result validates the generated sample on that machine;
it is not a general Urdu OCR accuracy benchmark.


## Currency-agent configuration contract

The Currency Agent accepts `input_size` only as a positive integer; booleans,
numeric strings and fractional values are rejected rather than coerced. Model
path resolution and estimator construction failures are returned as a
`needs_setup` result with exception messages and local paths omitted. A
successful setup only means the adapter was constructed; ONNX metadata checks
and supervised denomination evaluation remain required before deployment.


## Specialist runtime failure contract

OCR, object-detection, currency and speech adapters are called inside
specialist-level exception boundaries. An adapter exception returns a
structured `error` result with a bounded context and exception type; raw
exception messages, recognized text and local paths are not copied into the
summary. Currency results additionally require a mapping with a non-empty
denomination and a finite-range confidence value from 0 through 1. A failed
task must not be interpreted as a negative detection, verified banknote or
completed speech playback. The coordinator remains a second defensive
boundary for unexpected agent-level failures.
