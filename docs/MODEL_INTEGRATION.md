# Model integration contract

This document defines the current adapter contracts. It does not certify any model's accuracy.

## Currency classification (ONNX)

The `CurrencyEstimator` expects a single-image ONNX classifier:
- Input: float32 NCHW tensor, shape `[1, 3, 224, 224]` by default.
- Preprocessing: resize to 224x224, RGB channel order, pixel values divided by 255.
- Output: one vector with exactly one score per supplied label.
- Labels: pass the exact class order used during training, for example `["10", "20", "50"]` only if those are the model's actual classes.
- Output is a denomination estimate with a softmax-derived score. Calibration and validation are still required.

Do not use this adapter to authenticate notes, identify counterfeit currency, or make financial decisions. Do not invent class order from a model filename.

## Object detection (SSD-style ONNX)

The `ObjectDetector` currently expects a single-image SSD output shaped as rows of:
`[image_id, class_id, confidence, x1, y1, x2, y2]`.

Coordinates are normalized to 0..1. The adapter reports an image-relative horizontal zone only. It does not infer physical distance, collision risk, route safety, or whether an object is reachable.

Model input conventions vary. Confirm the exact model's required resize, mean, scale, channel order, and output tensor layout before use. The current default preprocessing is only a compatibility starting point for a matching SSD model, not a universal ONNX convention.

## Printed English OCR

`PrintedEnglishOCR` uses the local Tesseract executable through pytesseract. The Python package alone is insufficient; install Tesseract separately and verify `tesseract --version`. Scope is clear printed English only.

## Offline speech

`SpeechOutput` uses pyttsx3 and voices installed on the host operating system. Enumerate voices with `SpeechOutput.available_voices()`. Urdu output is only available if a suitable local Urdu voice exists and has been tested. No internet audio generation is performed by this adapter.

## Validation gate before exhibition

For every model/backend:
1. Record its source, license, version, input preprocessing and output schema.
2. Test on held-out examples representative of local lighting, camera angles and note/object variations.
3. Record precision/recall or task-specific error rates and known failure cases.
4. Test with the network disabled if offline operation is claimed.
5. Keep confidence/uncertainty visible; do not turn uncertain results into authoritative spoken instructions.
6. Never use NOORIX as a replacement for established mobility aids or human guidance.
