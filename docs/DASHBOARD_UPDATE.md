# Dashboard update

The desktop dashboard now uses a dark, high-contrast layout with a navigation rail,
readiness cards, a large live camera panel, quick actions, and a dedicated result
area. Camera frames are converted through Pillow's `ImageTk.PhotoImage` instead
of passing encoded PNG bytes directly to Tk's `PhotoImage`.

## Run

```bash
python -m pip install -e ".[dev,camera,speech,ocr,onnx]"
python -m noorix --check
python -m noorix --gui
```

The GUI is a research prototype. OCR still requires the Tesseract executable,
and object/currency analysis requires compatible, validated ONNX models and
matching labels in `config.json`. Speech depends on an installed local voice.
