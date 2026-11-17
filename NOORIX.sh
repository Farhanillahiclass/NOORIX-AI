#!/usr/bin/env bash
# One-command Windows Git Bash launcher for the local NOORIX-AI desktop app.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=============================================="
echo "        NOORIX-AI | LOCAL DESKTOP START"
echo "=============================================="

# Prefer the currently activated Conda/virtual environment Python.
if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] Python was not found on PATH."
  echo "Activate the noorix-ai Conda environment, then run: bash NOORIX.sh"
  exit 1
fi

PYTHON_BIN="$(command -v python)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("NOORIX-AI requires Python 3.11 or newer. Found: " + sys.version.split()[0])
print("Python:", sys.executable)
print("Version:", sys.version.split()[0])
PY

# Use the user's extracted local model package when it exists.
if [[ -d "/d/NOORIX_MODELS/EXTRACTED/models" ]]; then
  export NOORIX_MODELS_DIR="/d/NOORIX_MODELS/EXTRACTED"
elif [[ -d "$ROOT/models" ]]; then
  export NOORIX_MODELS_DIR="$ROOT"
fi
if [[ -n "${NOORIX_MODELS_DIR:-}" ]]; then
  echo "[OK] Model package: $NOORIX_MODELS_DIR"
else
  echo "[WARN] External model package not found. Configured model checks may be skipped."
fi

# Preserve the user's existing config. Create a starter config only if absent.
if [[ ! -f "$ROOT/config.json" ]]; then
  cp "$ROOT/config.example.json" "$ROOT/config.json"
  echo "[INFO] Created config.json from config.example.json."
fi
CONFIG="$ROOT/config.json"

# Install the local package without reaching the network when runtime extras
# are already present. If something is missing, install the declared extras.
if python - <<'PY'
modules = ("cv2", "PIL", "numpy", "torch", "torchvision", "pytesseract",
           "pyttsx3", "transformers", "soundfile", "scipy", "onnxruntime")
missing = []
for name in modules:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    print("Missing Python modules:", ", ".join(missing))
    raise SystemExit(1)
print("All configured Python runtime modules are importable.")
PY
then
  python -m pip install -e . --no-deps
else
  echo "[INFO] Installing NOORIX optional runtime dependencies."
  echo "[INFO] First-time installation may require an internet connection."
  python -m pip install -e ".[all,pytorch-currency,asr-whisper]"
fi

if ! python -c "import tkinter" >/dev/null 2>&1; then
  echo "[ERROR] Tkinter is unavailable in this Python installation."
  echo "Install a Python distribution that includes Tcl/Tk, then retry."
  exit 1
fi

if command -v tesseract >/dev/null 2>&1; then
  echo "[OK] Tesseract: $(tesseract --version 2>&1 | head -n 1)"
else
  echo "[WARN] Tesseract executable not found. OCR will be unavailable until installed."
fi

echo
echo "[INFO] Checking NOORIX local capabilities..."
python -m noorix --check --config "$CONFIG"

echo
echo "[INFO] Launching NOORIX-AI desktop dashboard..."
echo "[INFO] Open Testing Center inside the dashboard to run all model smoke tests."
exec python -m noorix --gui --config "$CONFIG"
