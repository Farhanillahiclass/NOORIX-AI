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

1. Python version requirement (3.11–3.13; aligned with the current CI matrix).
2. Bytecode compilation for src/, tests/ and scripts/.
3. The complete pytest suite (python -m pytest -q).
4. Installed Python dependency consistency (python -m pip check).
5. Offline dependency declaration audit (base dependencies, extras, installed-version compatibility when packaging is available, and version-bound review), with timestamped reports under reports/dependencies/.
6. Installed package capability check (python -m noorix --check), which does not open the camera or load model weights.
7. JSON syntax and supported semantic constraints in root config.example*.json files; symlinked config files and label-map path components are rejected.
8. Basic nbformat-4 JSON structure for project Jupyter notebooks, with a 64 MiB per-file inspection limit and bounded reads.
9. SHA-256 verification for every imported model package with an integrity manifest. The validation runner rejects symlinked components in the imported-model path and caps its preliminary manifest inspection at 16 MiB before invoking the separately bounded verifier. Missing inventories on older imports are reported as warnings.
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

Existing reports are not overwritten. Timestamp collisions create a fresh suffixed report directory, and report files are opened exclusively rather than overwritten. The output directory is rechecked for symlink components before directory creation. These checks reduce accidental overwrites but cannot eliminate every concurrent filesystem race. Validation report creation rejects symlinked components in the reports/validation output path before writing. This reduces unintended writes through linked directories but cannot eliminate concurrent filesystem mutation races. Captured subprocess output redacts the project-root and home-directory prefixes before saving, and command-start failures report only the exception type. Reports can still contain other diagnostic details; review them before sharing. The script excludes report folders,
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

If the project test and validation dependencies are not installed in the active environment:

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
`reports/benchmarks/`. The benchmark rejects symlinked components in its report
output path and refuses invalid cold-start timing values. This reduces unsafe
writes through linked directories but cannot eliminate concurrent filesystem
mutation races. Use the same image, model, configuration, power mode, and
hardware when comparing runs.

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

The checker rejects a wheelhouse path containing symlinks and rejects symlinked or non-regular wheel entries before invoking pip. It requires the NOORIX project wheel, confirms that the active pip supports `install --dry-run`, and asks pip to resolve the
selected extras using `--no-index` and `--dry-run`. These checks reduce unsafe path-following, but do not eliminate filesystem time-of-check/time-of-use races if another process changes the wheelhouse concurrently. A successful check means
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


## Declared dependency review

Run the offline metadata report independently with `python scripts/dependency_audit.py`. It lists every optional extra, checks local installed-version compatibility when packaging is installed, and reviews upper version bounds. Use `--strict-upper-bounds` only when a release policy explicitly requires that gate. This report does not query vulnerability databases or replace a reviewed lockfile. See [Dependency Audit](DEPENDENCY_AUDIT.md).


Notebook discovery does not follow symlinked directories. Inaccessible scan locations are reported as failures instead of silently producing a partial inventory.
