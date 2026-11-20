# NOORIX AI — Production Readiness Gates

**Status: NOT PRODUCTION READY.** This document is the release gate for moving
NOORIX from a research prototype to a responsibly deployable accessibility
assistant. A green unit-test suite is necessary but does not establish product
safety, recognition accuracy, or production readiness.

## What "100% ready" means

"100%" means every mandatory release gate below has current, reviewable evidence
and an owner has accepted the residual risks. It does **not** mean 100% OCR or
object-detection accuracy, zero defects, or guaranteed safe navigation. Those
claims are not realistic and must never be implied.

NOORIX must not be marketed or used as a replacement for a white cane, guide
dog, trained mobility support, or a person's own judgement. Image-relative
object positions are not distance measurements, collision detection, or route
guidance. Currency classification is not banknote authentication.

## Current evidence snapshot

Evidence recorded from the project branch and local test reports as of
2026-11-19:

| Area | Current evidence | Release status |
|---|---|---|
| Automated tests | 126 passed, 1 skipped, 1 existing ZIP-fixture warning on the developer's Windows environment | Partial |
| English OCR runtime | Synthetic high-contrast sample passed through Tesseract, OCR Agent, coordinator and Safety Agent | Partial |
| Urdu OCR runtime | Synthetic Urdu sample passed on one Windows machine with Tesseract `urd`, Pillow RAQM and Tahoma | Partial |
| Object detection | YOLOv8 COCO-80 ONNX adapter and local inference smoke checks exist; strict canonical label order and ONNX input/output metadata checks have been added, but new regression tests await the combined local run | Partial |
| Model import security | ZIP extraction limits, traversal/collision checks, privacy-minimized manifests and SHA-256 inventories/verifier added; regression tests await the combined local run | Partial |
| Real-world recognition quality | No documented representative held-out benchmark or acceptance report | Blocked |
| Target-device performance | A local single-image detector latency benchmark tool now records cold start, warm-run percentiles, throughput, platform and runtime versions; no representative target-device latency/RAM acceptance report exists yet | Blocked |
| Safety validation | Conservative warnings and explicit product limitations are documented; camera open/read failures stop preview and release handling now records only a release exception type | Partial |
| Diagnostic privacy | Agent and GUI exception summaries now omit raw exception messages to reduce accidental exposure of OCR text and local paths; background UI callbacks are guarded during shutdown; broader UI/console logging review remains outstanding | Partial |
| Model compatibility | YOLOv8 runtime now checks the one-input/one-output tensor contract and canonical COCO-80 label order before inference; regression coverage is pending the combined local run | Partial |
| Dependency policy | Added an offline declaration audit for optional extras, local installed-version presence, specifier compatibility checks when `packaging` is installed, and upper-bound review; no vulnerability database, lockfile or license scan is performed | Partial |\n| Installation / packaging | Added a documented platform-specific wheelhouse build and offline pip dry-run resolver; no wheelhouse has yet been built or validated on a clean offline target | Blocked |
| Licensing | Project license not selected; model and dataset redistribution rights need review | Blocked |
| Continuous integration | Historical PR #25 run passed Ubuntu Python 3.11/3.12/3.13 and Windows Python 3.12 (4/4 jobs); hosted runs are now paused to conserve the exhausted Actions allowance | Partial |

This table must be updated only when evidence is available. A skipped test is
not a passing runtime validation.

## Mandatory release gates

### 1. Build, compatibility and CI

- [ ] CI passes on every supported Python version and operating system.
- [ ] Supported OS/Python combinations are explicitly documented and tested.

Historical hosted evidence: PR #25 commit 9df3d0a passed Ubuntu Python 3.11/3.12/3.13 and Windows Python 3.12 (4/4 jobs). Hosted workflows have since been paused to avoid additional Actions usage. The new local validation runner on the current feature branch has not yet been executed; its full result must be recorded after the planned combined Windows validation.
- [ ] Clean installation succeeds from a fresh environment without relying on
  developer-machine state.
- [ ] Full test suite passes; every skip is understood and intentional.
- [ ] Static checks, dependency vulnerability review and secret scanning are
  configured and their findings triaged.
- [ ] Release artifacts can be built reproducibly and verified.

### 2. Model and recognition quality

- [ ] Publish an evaluation protocol, dataset provenance, split and model
  version for each recognition feature.
- [ ] OCR is evaluated on representative printed English and Urdu documents,
  including varied fonts, sizes, blur, perspective, lighting and contrast.
- [ ] OCR reports character/word error rates with sample counts and known
  limitations; no accuracy claims are made from synthetic samples alone.
- [ ] Object detection is evaluated against labeled held-out data with
  documented precision, recall and mAP where applicable.
- [ ] Currency estimation is evaluated on representative, lawfully sourced
  notes and lighting conditions; it is never described as authentication.
- [ ] Confidence thresholds and failure/abstention behavior are calibrated and
  documented.
- [ ] Model files, preprocessing, output contracts and label maps are versioned
  and integrity-checked.

### 3. Accessibility and safety

- [ ] All user-facing flows clearly state what the system can and cannot do.
- [ ] No feature implies safe obstacle avoidance, distance estimation or
  navigation unless separately engineered and independently validated.
- [ ] Missing models, camera errors, OCR uncertainty and low-quality frames
  produce understandable failure states rather than fabricated results.
- [ ] Speech output does not conceal errors or overstate confidence.
- [ ] Accessibility review covers keyboard operation, readable contrast,
  scalable text, screen-reader labels and understandable error messages.
- [ ] Any field evaluation is supervised, consent-based and designed with
  qualified accessibility specialists; NOORIX is not the sole mobility aid.

### 4. Privacy and security

- [ ] Camera frames and OCR text have documented collection, processing,
  retention and deletion behavior.
- [ ] No image, transcript, telemetry or diagnostic data leaves the device
  without clear disclosure and explicit user control.
- [ ] Logs exclude image contents, OCR text, credentials and unnecessary
  personal data by default.
- [ ] Configuration and model paths are validated; malformed or hostile files
  fail safely.
- [ ] Dependencies are pinned or constrained with a reviewed lock/release
  process; vulnerability findings are triaged.
- [ ] Secret scanning and a release-history review are completed before any
  repository visibility or binary distribution change.

### 5. Performance and reliability on target devices

- [ ] Define supported minimum CPU, RAM, camera and storage requirements.
- [ ] Measure cold start, camera frame rate, per-agent latency, memory use and
  sustained-session behavior on the actual supported hardware.
- [ ] Set product acceptance thresholds before benchmarking; do not invent
  performance claims from a few manual runs.
- [ ] Verify graceful behavior when the camera disconnects, models are missing,
  Tesseract is unavailable, or memory is constrained.
- [ ] Verify repeated start/stop and long-session cleanup without leaked camera
  handles or unbounded memory growth.

### 6. Distribution, support and legal readiness

- [ ] Select and publish a project software license after reviewing all code
  contributions and third-party components.
- [ ] Review Ultralytics/YOLOv8 AGPL-3.0 obligations and any alternative
  commercial licensing before redistribution or embedding.
- [ ] Verify rights and terms for every bundled model, label set, dataset,
  font, icon and other asset.
- [ ] Provide a clean installer or documented reproducible deployment process,
  uninstall steps and offline dependency/model provisioning.
- [ ] Provide versioning, release notes, support policy, known-issues register
  and rollback procedure.
- [ ] Complete a supervised acceptance test on every supported deployment
  configuration.

## Release decision

A release owner must review every mandatory gate, attach evidence, record
accepted residual risks, and explicitly approve the release. Any unresolved
safety, privacy, licensing, critical security or recognition-quality blocker
means **NO RELEASE**.

The project is not production-ready until all mandatory gates are checked with
evidence. The current project status remains **research prototype / supervised
evaluation only**.
