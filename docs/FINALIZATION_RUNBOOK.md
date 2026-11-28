# NOORIX Finalization Runbook

This runbook is the controlled path from the current development branch to a release decision. It deliberately separates code completion, automated validation, real-device evidence, and release approval.

## Current development constraints

- Work on `feature/local-validation-runner` through PR #26.
- Keep PR #26 open and in draft until the agreed final local validation and review are complete.
- Do not merge automatically.
- Do not dispatch GitHub Actions while the repository owner's included Actions allowance is exhausted.
- Do not commit local reports, model weights, camera frames, OCR samples, secrets, or machine-specific configuration.
- Do not claim production readiness based only on unit tests or a successful validation-runner exit code.

## Phase A — Finish implementation

Before asking for the final local run:

1. Review the current PR diff and all newly added tests for consistency with their modules and supported Python versions.
2. Review startup, model loading/reloading, agent orchestration, GUI callbacks, and shutdown paths for controlled failures.
3. Review camera, OCR, speech, and model diagnostics for private content and terminal-control injection.
4. Review importer, integrity verifier, and configured model paths for path traversal, symlink/junction handling, resource bounds, and non-destructive failure behavior.
5. Check documentation against the actual implemented behavior. Mark unmeasured recognition quality and device performance as not measured.
6. Keep tests and benchmarks unexecuted during this phase, as agreed. Do not describe unexecuted tests as passing.

## Phase B — Prepare the local machine

After implementation is declared feature-complete:

1. Back up any uncommitted local work and user-supplied model files.
2. Open Git Bash in the existing NOORIX repository.
3. Confirm the repository is on `feature/local-validation-runner` and the working tree has no unexpected changes.
4. Fetch and pull the latest branch commit. Do not reset, clean, or overwrite local files.
5. Activate the project's intended virtual environment and confirm its Python version is one of 3.11, 3.12, or 3.13.
6. Do not install or upgrade dependencies blindly; preserve the existing environment and review any installation requirement first.

## Phase C — One combined automated validation

Run only the agreed combined command from the repository root:

```bash
python scripts/run_full_validation.py
```

The runner performs compilation, the full pytest suite, pip consistency checks, offline dependency audit, package capability check, config/notebook validation, imported-model integrity checks, optional OCR inventory, and optional local wheelhouse resolution.

Record its exit code and generated report locations. If it fails, preserve the report, fix the underlying issue on the development branch, and repeat the combined command only after the correction. Never edit generated reports to make a failure appear successful.

## Phase D — Real-device and recognition evidence

A green automated run does not establish model quality or safe accessibility performance. Complete the following separately with permitted, representative data and supervised evaluation:

- Object detection: held-out precision/recall, mAP50, mAP50-95, class-wise errors, and dense-scene behavior.
- English and Urdu OCR: CER/WER, language-specific failure analysis, and actual supported Tesseract/font configuration.
- Currency estimation: per-denomination confusion matrix, abstention behavior, and false-estimate review. Do not claim banknote authentication.
- Target hardware: cold start, warm P50/P95 latency, RAM, sustained thermal behavior, and reliability.
- Offline deployment: clean-machine installation and operation with networking disabled.
- Accessibility: supervised evaluation of the intended use, uncertainty messages, and failure states.

Use `docs/RECOGNITION_EVALUATION_PROTOCOL.md` and `docs/FINAL_LOCAL_RELEASE_EVIDENCE_RECORD.md` to record evidence. Do not use NOORIX as the sole mobility aid during testing.

## Phase E — Release review

Before a release candidate is considered:

1. Review all local validation reports for private paths or data before sharing.
2. Complete model, dataset, dependency, and project license reviews.
3. Resolve all safety-critical, privacy, security, and legal blockers.
4. Complete the evidence record and document residual risks and deployment scope.
5. Request human review of PR #26.
6. Only after the owner explicitly approves, merge the PR and prepare a separately reviewed release artifact.

## Stop / no-go conditions

Stop the release process if the validation runner fails; model integrity cannot be established; a critical dependency or license question is unresolved; a model failure can be mistaken for a valid recognition result; private camera/OCR data is exposed; offline recovery is unverified for the intended deployment; or supervised accessibility review is missing.

A release may be described only at the evidence level actually achieved: research prototype, internally validated build, supervised limited pilot, or independently reviewed release. Never claim “100% accurate” or “100% production-ready.”
