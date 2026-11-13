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
- [x] Add local printed-English OCR adapter (Tesseract executable remains an external requirement).
- [x] Add local speech adapter (available language voices depend on the host OS).
- [x] Add ONNX currency and SSD-style object-detection adapter interfaces.
- [x] Add configuration template for model paths and exact class labels.
- [ ] Supply appropriately licensed, trained models and validate their preprocessing/class maps.

## Milestone 3 — Agent orchestration\n- [x] Add typed agent results and explicit readiness/error states.\n- [x] Add OCR, object, currency, speech, safety and system-health agents.\n- [x] Add deterministic coordinator and vision pipeline.\n- [x] Add desktop Agent Console and system-health view.\n- [x] Add unit tests for agent contracts and failure handling.\n\n## Milestone 4 — Evaluation
- [ ] Create held-out test datasets representative of local lighting, camera angles and note/object variations.
- [ ] Measure per-class performance, uncertainty calibration and failure cases.
- [ ] Conduct supervised usability testing with informed consent.
- [ ] Confirm Urdu voice availability and intelligibility on target hardware.

## Milestone 5 — Offline packaging
- [ ] Package Tesseract, voice dependencies and all required model weights.
- [ ] Verify complete operation with internet disabled.
- [ ] Build and test Windows desktop distribution.
- [ ] Evaluate separate Android and macOS implementation/build paths.

No recognition model or product milestone is marked complete without a compatible artifact and reproducible test evidence.
