# Final Local Release Evidence Record

Use this record after the unified local validation command has completed. Keep the generated JSON and Markdown reports in the local `reports/` directory; do not commit machine-specific reports, private OCR samples, camera frames, model weights, or personal paths.

## 1. Run identity

- Date/time and timezone:
- Operator:
- Operating system and version:
- Python version (must be 3.11, 3.12, or 3.13):
- Git commit:
- NOORIX branch:
- Hardware / CPU:
- Available RAM:
- Power mode and thermal conditions:
- Validation report directory:

## 2. Unified automated validation

Run from the repository root in the intended virtual environment:

```bash
python scripts/run_full_validation.py
```

Record the final process exit code and report paths:

- Exit code:
- JSON report:
- Markdown report:
- Failed checks:
- Skipped checks and reasons:
- Warnings requiring review:

A zero exit code means only that the checks implemented by this runner passed. It is not a production-readiness certificate.

## 3. Runtime and model inventory

For every model used, record:

| Module | Model filename / version | Source and provenance | SHA-256 verified | License reviewed | Runtime result |
|---|---|---|---|---|---|
| Object detection | | | | | |
| Currency estimation | | | | | |
| OCR / Tesseract data | | | | | |
| Other | | | | | |

Do not record access tokens, private URLs, or sensitive user data. A valid hash proves file integrity against the manifest, not model authenticity, safety, accuracy, or licensing.

## 4. Recognition quality evidence

Use a held-out, representative dataset with documented permissions and ground truth. Do not tune on the evaluation split.

- Dataset provenance / permission:
- Split definition and sample counts:
- Object detection: mAP50, mAP50-95, per-class precision/recall, false-positive and false-negative review:
- English OCR: CER/WER and representative failure categories:
- Urdu OCR: CER/WER and representative failure categories:
- Currency estimation: per-denomination confusion matrix and abstention/error review:
- Acceptance thresholds approved before evaluation:
- Evaluator and review date:

Attach or reference the detailed protocol in `docs/RECOGNITION_EVALUATION_PROTOCOL.md`. If a metric has not been measured, mark it **NOT MEASURED** rather than estimating it.

## 5. Target-device performance and reliability

Run the documented CPU benchmark with a representative local image and retain its generated report locally.

- Device / CPU:
- Model and input dimensions:
- Warm-up count and measured iterations:
- Cold-start latency:
- Warm P50 / median latency:
- Warm P95 latency:
- Peak RAM / memory measurement method:
- Thermal / sustained-run observations:
- Camera and GUI lifecycle observations:
- Long-run duration and interruptions:

The single-image benchmark does not measure end-to-end camera, OCR, speech, memory, thermal, or safety performance. Record those separately.

## 6. Offline installation and recovery

- Clean Windows account or clean machine used:
- Python installation source/version:
- Wheelhouse contents and provenance:
- Installation completed with network disabled:
- GUI starts with network disabled:
- Required model and language assets available offline:
- Missing/corrupt model produces a safe unavailable state:
- Upgrade, backup and rollback procedure verified:
- Logs and reports reviewed for private data:

## 7. Accessibility, privacy and safety review

- Supervised accessibility evaluator:
- Tasks and environment:
- User feedback and limitations:
- Clear messaging that NOORIX is not a replacement for a white cane, guide dog, trained mobility support, or independent judgement:
- Camera data remains session-local unless the user explicitly exports it:
- No unapproved image/OCR content in logs or reports:
- Failure, uncertainty and unavailable-model states are understandable:
- No safety-critical navigation claims:
- Open risks and mitigations:

Do not test the prototype as a sole means of mobility or in uncontrolled hazardous environments.

## 8. Security and legal review

- Dependency and vulnerability review (the offline dependency audit is not a vulnerability scanner):
- Model and dataset provenance:
- Model/dataset licenses and redistribution terms:
- Project license decision:
- Importer adversarial archive review:
- Integrity manifest review:
- Secrets and personal-data scan:
- Security reviewer and date:
- Unresolved findings:

## 9. Release decision

Choose one:

- [ ] **NO-GO** — one or more mandatory gates are missing or blocked.
- [ ] **CONDITIONAL / LIMITED PILOT** — all safety-critical gates pass; explicitly documented non-critical limitations remain, with owner approval and restricted deployment.
- [ ] **GO FOR REVIEW** — evidence is complete and ready for independent release review. This is not automatic authorization to publish.

Release owner:
Decision date:
Approved deployment scope:
Accepted residual risks:
Required follow-up date:

## Mandatory no-go conditions

Do not release if any of the following remain unresolved: safety-critical recognition failures; unsafe failure behavior; privacy or security blockers; unclear model/dataset rights; inability to install or recover the intended offline deployment; or missing supervised accessibility review for the intended use.

Never describe the product as “100% accurate.” Report measured performance, dataset scope, known limitations, and residual risks.
