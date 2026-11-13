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

The current `ObjectDetector` supports only a **single post-processed output tensor** containing rows in this exact order:

`[image_id, class_id, confidence, x1, y1, x2, y2]`

The four box coordinates must be normalized to 0..1. Invalid/non-finite rows, out-of-range classes, and invalid boxes are ignored. The adapter reports an image-relative horizontal zone only. It does not infer physical distance, collision risk, route safety, or whether an object is reachable.

**Raw multi-head SSD models are not supported by this adapter yet.** If the ONNX model exposes multiple output tensors, the adapter raises `NotImplementedError` instead of guessing anchor generation, box coding, score activation, class mapping, or non-maximum suppression. A model-specific decoder must be independently verified against the exact model export before enabling it.

The bundled research model `ssd-mobilenet-v2.onnx` has multiple raw output heads and therefore does not currently satisfy this adapter contract. Its graph loading and CPU inference smoke test do not establish that detections are decoded correctly or that the model is accurate.

The current generic preprocessing path is not a verified preprocessing implementation for every SSD model. Confirm the exact model's resize interpolation, mean, scale, channel order, layout, and output schema before use. Do not infer these from the model filename.

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
