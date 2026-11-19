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
2. Bytecode compilation for src/ and tests/.
3. The complete pytest suite (python -m pytest -q).
4. Installed Python dependency consistency (python -m pip check).
5. JSON syntax of root config.example*.json files.
6. Basic nbformat-4 JSON structure for project Jupyter notebooks.
7. Optional Tesseract executable and installed language inventory.

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
