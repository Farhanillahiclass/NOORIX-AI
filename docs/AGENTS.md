# NOORIX AI Agent Architecture

NOORIX uses a deterministic local specialist-agent design. "Agent" means a
bounded software component with a defined input/output contract; it does not
imply an autonomous LLM or unrestricted tool access.

## Specialist agents

| Agent | Input | Responsibility |
|---|---|---|
| OCR | Camera frame | Read clear printed English through Tesseract |
| Object Detection | Camera frame | Run configured SSD-style ONNX model; report image-relative positions |
| Currency | Camera frame | Run configured ONNX classifier; estimate denomination only |
| Speech | Text | Speak through an installed local system voice |
| Safety | Specialist results | Attach conservative warnings and disclose failed tasks |
| System Health | Local configuration | Report adapter, executable and model-file readiness |\n| Frame Quality | Camera frame | Check dimensions, brightness and blur; report image suitability only |\n| Voice Command | Transcribed text | Map a small English/Roman Urdu/Urdu allowlist to an intent; require caller confirmation |

## Coordinator

AgentCoordinator registers agents by unique name and executes only the requested
agent. It converts runtime exceptions into structured error results. The vision
pipeline runs object and currency agents, then passes their results to the Safety
Agent. No agent downloads files, sends frames to a server, or saves images.

## Honest readiness states

- ready: configured and available to attempt a task.
- running: reserved for UI progress indication.
- completed: task returned a result; this is not a claim of model accuracy.
- needs_setup: required frame, model, labels, executable or voice is missing.
- error: task failed and the UI should show the reason.

A model adapter is not production-ready until exact preprocessing, output shape,
label order and held-out evaluation are verified.

## Safety boundary

Image-relative left/centre/right is not distance estimation, obstacle avoidance
or navigation. Currency output is not banknote authentication. NOORIX must never
be presented as a replacement for a white cane, guide dog or trained mobility
support.
