# NOORIX-AI — One-command local validation

Hosted GitHub Actions are paused while the account's included Actions allowance
is exhausted. Do not manually dispatch the workflow while this policy is active.
Development can continue on GitHub without running hosted tests; final validation
is performed together on the developer's Windows machine.

## Command

From the repository root, using the project's activated Python environment:

~~~bash
python scripts/run_full_validation.py
~~~

The same command works in Git Bash and PowerShell. It does not call GitHub,
download datasets, access a camera, or send local images or OCR text anywhere.

## Checks performed

1. Python version requirement (3.11+).
2. Bytecode compilation for src/, tests/ and scripts/.
3. The complete pytest suite (python -m pytest -q).
4. Installed Python dependency consistency (python -m pip check).\n5. Offline dependency declaration audit (base dependencies, extras, local installed versions and version-bound review), with timestamped reports under reports/dependencies/.
6. Installed package capability check (python -m noorix --check), which does not open the camera or load model weights.
7. JSON syntax and supported semantic constraints in root config.example*.json files.
8. Basic nbformat-4 JSON structure for project Jupyter notebooks.
9. SHA-256 verification for every imported model package with an integrity manifest. Missing inventories on older imports are reported as warnings.
10. Optional Tesseract executable and installed language inventory.
11. If a local wheelhouse/ directory exists, resolve all runtime and development extras from it using pip dry-run with network access disabled; otherwise report INFO.

Tesseract is informational: if it is missing, OCR runtime tests may skip according
to their own test conditions. A warning is not counted as a passing test.
A failed required check makes the command exit with status 1.

## Reports

Each run writes two files to a new timestamped folder under
reports/validation/:

- validation-report.md — readable summary and command output.
- validation-report.json — machine-readable result details.

Existing reports are not overwritten. The script excludes report folders,
virtual environments, Git metadata, caches and build output when searching for
notebooks. It only parses example configuration files; it does not read the
user's local config.json.

## What this command does not certify

A green local run is not proof of OCR accuracy, object-detection quality,
banknote authenticity, safe navigation, privacy compliance, accessibility
acceptance, license rights, clean-machine installation or production readiness.
Those gates require separate documented evidence and, where appropriate,
supervised human evaluation on the target device.

## Dependency setup if needed

If pytest is not installed in the active environment:

~~~bash
python -m pip install -e ".[all,dev]"
~~~

This installs Python dependencies from the repository. Tesseract itself and
trained language data are separate system installations.


## Optional CPU object-detection benchmark

After configuring a compatible model, run a repeatable single-image timing
sample:

```bash
python scripts/benchmark_runtime.py --config config.json --image tests/assets/coco_sample.jpg --iterations 30 --warmup 5
```

The script records cold first-inference time, warm-run mean/median/P95/min/max,
approximate throughput, runtime/platform metadata, and the number of detections
from the final run. It writes JSON and Markdown reports under
`reports/benchmarks/`. Use the same image, model, configuration, power mode,
and hardware when comparing runs.

This benchmark does not measure memory, camera capture, OCR, speech, thermal
throttling, recognition accuracy, or end-to-end user experience. A single image
cannot establish real-world performance or accessibility safety. Do not use
its FPS figure as a product claim.


## Privacy-safe GUI diagnostics

GUI error messages now show a short operation context and exception type only.
Raw exception text is not copied into user-facing status, dialogs or error
summaries, because library and driver messages can contain local paths or
content. Detailed diagnostics are intentionally limited; use the local
validation report and reproduce configuration problems without adding private
images or OCR text to logs.


## Offline installation bundle

Build a wheelhouse on an internet-connected machine using the **same operating
system, CPU architecture and Python version** as the offline target:

~~~bash
python -m pip install --upgrade pip
python -m pip wheel --wheel-dir wheelhouse ".[all,dev]"
~~~

Copy the repository and the complete `wheelhouse/` directory to the offline
machine. From the repository root, verify dependency resolution without
network access or installing anything:

~~~bash
python scripts/verify_offline_bundle.py --wheelhouse wheelhouse --extras all,dev
~~~

The checker requires the NOORIX project wheel and asks pip to resolve the
selected extras using `--no-index` and `--dry-run`. A successful check means
the local wheel files satisfy the current Python environment's dependency
resolver; it does not prove runtime functionality.

After reviewing a successful report (also included automatically by the final validation runner when wheelhouse/ exists), install from the local wheelhouse:

~~~bash
python -m pip install --no-index --find-links wheelhouse "noorix-ai[all,dev]"
~~~

Do not reuse a wheelhouse across different Python versions, operating systems
or CPU architectures without rebuilding and rechecking it. Native packages
such as OpenCV and ONNX Runtime are platform-specific. Tesseract executable
and language-data files are separate system assets and are not provided by
Python wheels. Model weights must be provisioned separately and their
licenses/integrity verified.
\n\n## Declared dependency review\n\nRun the offline metadata report independently with `python scripts/dependency_audit.py`. It lists every optional extra and checks whether dependency ranges have upper bounds. Use `--strict-upper-bounds` only when a release policy explicitly requires that gate. This report does not query vulnerability databases, verify installed versions against specifiers, or replace a reviewed lockfile. See [Dependency Audit](DEPENDENCY_AUDIT.md).\n