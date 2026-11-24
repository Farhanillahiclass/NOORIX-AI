# NOORIX-AI Recognition and Device Evaluation Protocol

## Purpose and status

This protocol defines the evidence required before making claims about object detection, printed-text OCR, or practical accessibility performance. It is an evaluation plan, not evidence that NOORIX currently meets any threshold. Do not describe the prototype as safe for independent navigation or as a replacement for a white cane, guide dog, trained mobility support, or human judgement.

## 1. Dataset governance

Before evaluation:

- Record the dataset owner, source URL or acquisition record, license/terms, permitted use, and redistribution restrictions.
- Keep training, tuning, and held-out test sets separate. Do not tune thresholds or preprocessing on the held-out test set.
- Split by source scene/document/session where possible, not merely by adjacent frames or crops. This reduces near-duplicate leakage.
- Record dataset version, collection date, label schema, exclusions, and class distribution.
- Avoid collecting identifiable personal data unless there is a documented lawful basis, necessity, consent/approval where required, retention policy, and access control.
- Keep private images and OCR text local. Evaluation reports should contain aggregate metrics and dataset identifiers, not raw images or extracted personal text.
- Use representative lighting, blur, occlusion, camera angles, printed fonts, page layouts, and Urdu/English language samples. Document gaps instead of implying coverage.

## 2. Object-detection evaluation

For every held-out image, preserve ground-truth boxes and class labels under a documented annotation guide. Report:

- Precision, recall, and F1 at the declared operating threshold.
- Per-class precision and recall, including support (number of ground-truth instances).
- mAP@0.50 and mAP@0.50:0.95 using a documented evaluator and class mapping.
- False positives and false negatives by scene condition.
- Confidence calibration or at least confidence-bin precision, where feasible.
- Results separately for each dataset split and relevant operating condition.

Record the exact model file SHA-256, export format, label-map version/hash, inference backend, preprocessing, input dimensions, confidence and NMS thresholds, runtime/library versions, and hardware. Do not compare metrics produced with different evaluator settings as if they were directly equivalent.

The current CPU timing script measures one supplied image repeatedly. It does not calculate detection accuracy or mAP and cannot replace this evaluation.

## 3. OCR evaluation

Create a manually verified ground-truth transcription for each held-out printed sample. Keep punctuation and normalization rules explicit, especially for Urdu joining, diacritics, whitespace, and mixed English/Urdu text.

Report separately for English and Urdu:

- Character error rate (CER).
- Word error rate (WER).
- Exact-line accuracy.
- Empty/failed extraction rate.
- Results by print size, font, contrast, blur, skew, page layout, and language mix.

State the normalization applied before scoring. Do not silently remove difficult samples or normalize away meaningful recognition errors. OCR quality on synthetic text images is a smoke test only; it is not representative real-world OCR evidence.

## 4. Target-device performance and reliability

Run on the actual supported target machine, with exact hardware and software versions recorded. Use fixed, disclosed power settings and allow the device to return to a comparable thermal state between runs.

Measure separately:

- Cold model initialization and first inference.
- Warm inference P50, P95, and maximum latency across a predeclared number of runs.
- Peak and steady-state memory use.
- Camera capture-to-result latency and frame-drop rate.
- OCR and speech latency, measured separately from object detection.
- Startup, shutdown, repeated start/stop, and long-session behavior.
- Behavior when optional models, Tesseract, camera devices, or language data are unavailable.

Repeat the measurements across multiple sessions. Record failures and outliers; do not report only the fastest run. A single-image FPS estimate is not end-to-end responsiveness.

## 5. Accessibility and safety evaluation

Conduct supervised usability sessions with appropriate accessibility expertise and informed participants. Define the intended task, environment, fallback behavior, and stop conditions before each session.

- Do not ask participants to rely on NOORIX for independent travel or obstacle avoidance.
- Do not test near roads, stairs, traffic, or other hazardous environments as a substitute for established mobility aids.
- Confirm that uncertain, stale, unavailable, or low-confidence results are clearly communicated.
- Verify that directional labels describe image position only and are not presented as physical distance or safe-route instructions.
- Record false reassurance, missed objects, confusing speech, latency, and failure recovery as safety-relevant findings.
- Maintain established mobility aids and trained support throughout evaluation.

## 6. Reporting template

Every evaluation report should include:

| Field | Required record |
|---|---|
| Evaluation ID/date | Unique identifier and date |
| Dataset | Version, source, license, split, sample counts |
| Model | Filename, SHA-256, provenance, license |
| Runtime | Backend, preprocessing, thresholds, library versions |
| Device | CPU, RAM, OS, Python, power/thermal conditions |
| Metrics | Per-task and per-language metrics with sample support |
| Failures | False positives/negatives, skipped samples, crashes |
| Limitations | Uncovered conditions and known bias |
| Reviewer | Evaluator and independent reviewer |
| Decision | Pass, fail, or inconclusive with rationale |

Do not invent universal pass thresholds after seeing the results. Define task-specific acceptance thresholds with relevant domain reviewers before running the held-out evaluation. Preserve raw evaluation artifacts securely under the dataset's terms, while keeping published reports aggregate and privacy-conscious.

## 7. Release gate

A release claim requires:

1. Reproducible held-out object-detection and OCR reports.
2. Target-device performance and reliability evidence.
3. Independent review of dataset/model provenance and licensing.
4. Supervised accessibility evaluation and documented failure handling.
5. Privacy/security review and approved data-retention behavior.
6. Explicit release-owner acceptance of residual risks.

Until these records exist and are reviewed, the correct status is **research prototype / supervised evaluation only**.
