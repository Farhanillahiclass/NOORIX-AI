# NOORIX-AI Architecture

## Product boundary

NOORIX-AI is an offline-first research prototype for camera-based accessibility assistance. It provides visual observations and printed-text extraction; it is not a certified assistive device, a navigation system, or a replacement for a white cane, guide dog, trained mobility support, or the user's own judgement.

The application runs locally. Camera frames are held in memory for the active session and are not saved automatically. Model weights, Tesseract and operating-system speech voices are provisioned separately.

## Runtime components

- `camera.py`: explicit webcam start/read/stop lifecycle and release handling.
- `gui.py`: Tk desktop interface, camera preview, manual and automatic analysis, status and result presentation.
- `app.py`: CLI readiness report, camera preview and desktop launch.
- `ocr.py`: local printed English/Urdu OCR through Tesseract.
- `objects.py`: post-processed SSD output adapter and backend factory.
- `ssd_mobilenet_v2.py`: model-specific raw-head SSD MobileNet V2 decoder with fixed anchor contract.
- `yolov8_onnx.py`: CPU ONNX Runtime adapter for standard COCO-80 YOLOv8 exports.
- `currency.py`: ONNX classifier interface for Pakistani banknote denomination estimation.
- `speech.py`: local speech output through installed operating-system voices.
- `agents/`: typed specialist results, deterministic coordinator, frame-quality checks, voice-intent parsing and conservative safety review.
- `paths.py`: configuration-relative path resolution, bounded JSON reading and regular-file/symlink/junction checks.
- `privacy.py`: bounded user-facing error summaries that omit raw exception messages and terminal controls.

## Main data flow

```text
Webcam
  -> in-memory frame
  -> frame-quality assessment
  -> selected OCR / object / currency agents
  -> structured AgentResult objects
  -> Safety Agent warnings
  -> GUI result summary
  -> optional local speech output
```

The coordinator is deterministic: it runs only registered agents explicitly requested by the caller. It does not independently choose actions, access the network, or perform hidden autonomous workflows. The Safety Agent attaches warnings; it does not prove an area is safe.

## Model adapter contracts

- Currency: one float32 NCHW input and one float32 score output, with one score per configured denomination label. Fixed dimensions and tensor counts are checked before the session is retained. Non-finite scores and score-count mismatches fail closed.
- SSD MobileNet V2: one float32 `[1,3,300,300]` input and 12 float32 output heads with shapes fixed to the verified 1,917-anchor layout. The model and anchor file must be supplied separately.
- Generic SSD adapter: one post-processed seven-column detection tensor only. It does not guess raw SSD decoding.
- YOLOv8: one float32 NCHW input and one raw COCO-80 output with 84 values per candidate. Candidate and per-class NMS limits bound CPU post-processing; the NMS cap may affect recall in dense scenes and has not been evaluated.

A model filename does not establish compatibility. Each model requires exact source, license, version, preprocessing, output schema, label order and held-out evaluation evidence.

## Local trust boundaries

- Runtime JSON and external object-label files are read through a bounded, duplicate-key-rejecting loader. Symlinked and Windows-junction path components are rejected; descriptor/path metadata comparisons detect observable file replacement or mutation.
- Model and SSD anchor paths must identify regular files with no symlink/junction components.
- Model ZIP archives are untrusted input. Import uses traversal, type, size, compression-ratio, collision, symlink and overwrite protections. Imported artifacts receive SHA-256 inventories.
- The integrity verifier hashes files without loading model formats and rejects unsafe manifest paths, duplicate JSON keys, symlinks/junctions, malformed inventories and observable file changes.
- User-facing error summaries omit raw exception text to reduce accidental exposure of local paths or OCR content.
- Filesystem checks reduce risk but cannot eliminate every concurrent filesystem race. Hash integrity does not prove provenance, licensing or model correctness.

## Validation and evidence

The repository provides one combined local command:

```bash
python scripts/run_full_validation.py
```

It compiles source, runs the test suite, checks installed dependency consistency, audits declared dependencies, checks example configuration and notebooks, verifies imported model manifests, inventories Tesseract, and optionally resolves a local wheelhouse with pip dry-run and `--no-index`.

The command creates timestamped JSON and Markdown reports under `reports/validation/`. It does not upload camera frames, OCR content, model weights or reports. Generated reports are ignored by Git and should be reviewed before sharing.

The optional CPU benchmark measures a single caller-supplied image; it does not measure accuracy, memory, thermal behavior, camera throughput, OCR, speech or end-to-end accessibility outcomes.

## Current limitations

- No representative held-out recognition dataset or accepted accuracy report exists.
- No target-device latency/RAM/thermal/reliability acceptance report exists.
- Urdu OCR requires external Tesseract language data; Urdu speech depends on installed local voices.
- Microphone speech-to-text is not implemented; voice commands currently parse text supplied by another component.
- Model weights are not bundled; the project license and third-party redistribution rights require review.
- Clean-machine offline installation and Windows distribution have not been validated.
- Supervised accessibility evaluation has not been completed.

See [Production Readiness Gates](PRODUCTION_READINESS.md), [Recognition and Device Evaluation Protocol](RECOGNITION_EVALUATION_PROTOCOL.md), and [Safety](SAFETY.md) before any public demonstration or release.
