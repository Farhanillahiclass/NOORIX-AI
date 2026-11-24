# Development Plan

## Milestone 0 — Repository scaffold
- [x] Initialize package layout and project documentation.
- [x] Add image-relative position helper and unit tests.
- [x] Add editable package installation and CI test workflow.

## Milestone 1 — Camera and desktop foundation
- [x] Add explicit webcam start/stop lifecycle and context-manager cleanup.
- [x] Add command-line camera preview with Q, Escape, window-close and Ctrl+C exits.
- [x] Add Tk desktop interface with camera controls and visible status.
- [x] Ensure frames are not saved automatically.

## Milestone 2 — Backend adapters
- [x] Add local printed English/Urdu OCR adapter (Tesseract executable and language data remain external requirements).
- [x] Add local speech adapter (available language voices depend on the host OS).
- [x] Add ONNX currency, model-specific SSD MobileNet V2, generic post-processed SSD and standard YOLOv8 COCO-80 object-detection adapters.
- [x] Add configuration template for model paths and exact class labels.
- [ ] Supply appropriately licensed, trained models and validate their preprocessing/class maps.

## Milestone 3 — Agent orchestration
- [x] Add typed agent results and explicit readiness/error states.
- [x] Add OCR, object, currency, speech, safety and system-health agents.
- [x] Add deterministic coordinator and vision pipeline.
- [x] Add desktop Agent Console and system-health view.
- [x] Add frame-quality checks for resolution, brightness and blur.
- [x] Add a confirmation-gated English/Roman Urdu/Urdu command-intent parser.
- [x] Add unit tests for agent contracts and failure handling (latest PR #26 regression additions are pending the final combined local run).

## Milestone 4 — Evaluation
- [x] Publish a held-out recognition and target-device evaluation protocol.
- [ ] Create held-out test datasets representative of local lighting, camera angles and note/object variations.
- [ ] Measure per-class performance, uncertainty calibration and failure cases.
- [ ] Conduct supervised usability testing with informed consent.
- [ ] Add and validate offline speech-to-text for microphone voice commands.
- [ ] Confirm Urdu voice availability and intelligibility on target hardware.

## Milestone 5 — Offline packaging
- [x] Add a local wheelhouse resolver using pip dry-run and `--no-index`; actual wheelhouse creation and clean offline installation remain pending.
- [ ] Package Tesseract, voice dependencies and all required model weights.
- [ ] Verify complete operation with internet disabled.
- [ ] Build and test Windows desktop distribution.
- [ ] Evaluate separate Android and macOS implementation/build paths.

No recognition model or product milestone is marked complete without a compatible artifact and reproducible test evidence.


## Milestone 6 — Production and trust-boundary hardening
- [x] Add a single local validation entry point with timestamped JSON/Markdown reports.
- [x] Add bounded, symlink/junction-aware JSON/config/model path checks.
- [x] Add safe ZIP import controls and SHA-256 model manifests/verifier.
- [x] Add offline dependency declaration and installed-version audit.
- [x] Add CPU detector timing-report tooling with explicit limitations.
- [x] Document release gates, safety boundaries, evaluation protocol and public-release checklist.
- [ ] Execute the final combined local validation on the supported Windows environment.
- [ ] Review every resulting failure/skip/warning and close blockers.
- [ ] Complete held-out recognition evaluation, target-device performance and supervised accessibility review.
- [ ] Complete legal/licensing review and clean offline distribution tests.

The checked items above indicate code or documentation has been added to the feature branch; they do not imply that the latest regression suite or real-world release gates have passed.
