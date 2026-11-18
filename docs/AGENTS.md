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
