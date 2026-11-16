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

## YOLOv8 ONNX object detection

The `YOLOv8ONNXDetector` is a separate adapter for standard Ultralytics YOLOv8 **Detect** exports. It does not reuse the SSD decoder.

Supported contract:
- Input: one BGR camera frame; letterboxed to 640x640 with padding value 114.
- Tensor: RGB, float32 NCHW, values scaled to 0..1.
- Output: exactly one raw detection tensor, either `[1, 4 + class_count, candidates]` or `[1, candidates, 4 + class_count]`.
- Each candidate contains `cx, cy, width, height`, followed by one class score per label. This is the standard YOLOv8 Detect layout without an objectness column.
- Class labels are read from a JSON list or a numeric-keyed object with contiguous indices starting at zero.
- Boxes are mapped back through the letterbox scale/padding, clipped to the original frame, then class-wise NMS is applied.

Configure `object_detection.backend` as `yolov8_onnx`, with `model_path`, `labels_path`, and an optional `confidence_threshold`. The example config points to the YOLOv8n package folder.

**Compatibility gate:** this adapter rejects multi-output, segmentation, pose, end-to-end/NMS, objectness-based, and other unrecognized output layouts. The Kaggle export notebook records a 640x640 input, but the actual ONNX file is not stored in this repository; therefore, CI tests validate the documented decoder contract with controlled tensors, not the user's particular ONNX artifact. Before relying on a downloaded export, verify its runtime output shape and compare detections against Ultralytics on the same sample images. YOLOv8n in the current package is COCO-pretrained unless a separate labelled custom-training run is documented. It is not a custom NOORIX dataset-trained model.

As with all NOORIX detections, image-relative positions are not distance estimates, obstacle avoidance, or safe-navigation instructions.

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

### Research notes for the bundled Kalray SSD MobileNet V2 model

The Hugging Face model card identifies this artifact as an ONNX conversion of a TensorFlow SSD MobileNet V2 model and links the TensorFlow Object Detection API COCO configuration. The model card declares Apache-2.0 licensing:
- Model card: https://huggingface.co/Kalray/ssd-mobilenet-v2
- Referenced TensorFlow configuration: https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config

That referenced configuration describes 90 foreground classes, a 300x300 fixed input, Faster R-CNN box-coder scales (y/x 10, height/width 5), sigmoid score conversion, and class-wise NMS configured with IoU 0.6 and up to 100 detections. TensorFlow's SSD postprocessor documents a background score at class index 0 and removes it when the model is configured with a background class.

These values are a **reference profile**, not yet a verified decoder specification for the checked-in ONNX graph. In particular, the inspected ONNX heads have channel counts consistent with 3 anchors per location on the 19x19 head and 6 on each remaining head (1,917 anchors total). The official TensorFlow helper's default reduced-lowest-layer path creates exactly three anchors per location on the first layer; the remaining layers create the configured aspect-ratio anchors plus one interpolated square anchor (six per location for this configuration). This matches the checked-in ONNX channel counts (3 anchors on the 19x19 head and 6 on each remaining head) and the 1,917-anchor total. This is strong structural consistency with the referenced configuration, but it does not by itself prove the exact exported anchor order or how the ONNX Conv channels are flattened into anchor/coordinate or anchor/class dimensions. Do not implement or enable decoding based on the reference profile alone.

Before implementing the decoder, verify the exported graph's exact feature-head order, channel-to-anchor/coordinate layout, effective anchor settings, class/background encoding, and box-coder behavior against the source TensorFlow checkpoint or a trusted, independently generated reference output. Record reproducible comparisons on known images. Until then, keep raw multi-head inference explicitly unsupported by `ObjectDetector`.

### Additional ONNX graph inspection findings

A local inspection of the pinned ONNX artifact recorded the following graph-level facts:

- The graph has one input named `FeatureExtractor/MobilenetV2/MobilenetV2/input:0` and twelve outputs.
- The twelve outputs are ordered as six pairs: `BoxPredictor_0` box then class, through `BoxPredictor_5` box then class.
- Each pair's box and class Conv nodes consume the same feature tensor for that head.
- The six feature-map resolutions are 19x19, 10x10, 5x5, 3x3, 2x2, and 1x1.
- Box Conv output channels are 12 on the first head and 24 on each later head. Class Conv output channels are 273 on the first head and 546 on each later head.
- The inspected prediction Conv nodes use 3x3 kernels, stride 1x1, padding [1, 1, 1, 1], dilation 1x1, and group 1. Each Conv has a weights and biases initializer.
- The ONNX graph has 175 nodes and 175 initializers; the inspected operation types are Add, Clip, Conv, and Mul. There is no visible Sigmoid, Softmax, or NMS node on the inspected graph.

The dimensions are consistent with 3 anchors per location on the 19x19 head and 6 anchors per location on each remaining head. The class channel dimensions are also consistent with 91 values per anchor (273 / 3 and 546 / 6). Together with the referenced COCO configuration's 90 foreground classes, this suggests a background-plus-90 class dimension, but the exact exported class index mapping has not yet been independently confirmed.

The TensorFlow reference generator documents that anchor order varies fastest by box specification, then width, then height, then feature-map index. For the reduced lowest layer, the three first-layer box specifications are (0.1, 1.0), (scale, 2.0), and (scale, 0.5); later layers use the configured aspect-ratio specifications followed by the interpolated square anchor. This describes the reference anchor generator's order only. It does **not** prove that the ONNX Conv channels use the same order when flattened into anchors and box/class values.

The graph inspection therefore strengthens the structural match but does not close the decoder verification gate. Still unverified: exact channel-to-anchor/coordinate mapping, exact class index mapping, box decoding against this export, and equivalence to a trusted reference output on known images. Keep raw multi-head SSD inference unsupported until those checks are completed.

### TensorFlow predictor-head source comparison

The referenced TensorFlow Object Detection API implementation provides an additional reference for the prediction-head layout:

- The standard `ConvolutionalBoxHead` emits `num_predictions_per_location * box_code_size` channels and reshapes the channels-last prediction tensor to `[batch, -1, 1, box_code_size]`. For the referenced SSD config, `box_code_size=4`.
- The standard `ConvolutionalClassHead` emits `num_predictions_per_location * num_class_slots` channels and reshapes to `[batch, -1, num_class_slots]`. The referenced COCO config has 90 foreground classes; the SSD meta-architecture prediction contract includes a background slot, yielding 91 class slots when configured with an explicit background class.
- In this reference implementation, the class head's own `apply_sigmoid_to_scores` option is false in the cited config. The meta-architecture's post-processing score-conversion function is configured separately as sigmoid. Therefore, class logits from the raw heads require the configured score conversion during post-processing; do not apply an arbitrary softmax.
- The source predictor implementation documents a channels-last reshape. The inspected ONNX graph exposes NCHW Conv tensors and does not contain the TensorFlow reshape nodes. A candidate interpretation is to transpose each ONNX head from NCHW to NHWC and then apply the corresponding anchor/slot reshape, but that remains a hypothesis until it is checked against the exact export or trusted reference outputs.

This source comparison narrows the expected TensorFlow reference layout, but does not prove that the particular Kalray ONNX export preserves the same channel packing and feature-map flattening. In particular, do not yet assume the exported Conv output channel index maps to a specific anchor and coordinate/class slot without numerical equivalence testing. The raw multi-head adapter remains unsupported.

### Pinned TensorFlow graph reference artifact

The exact Hugging Face revision pinned for the ONNX artifact is `9b8fd4f109071d88f3127424f2bf879bc8ae4fa4`. The repository tree at that same revision contains both `ssd-mobilenet-v2.onnx` and `ssd-mobilenet-v2.pb`. The TensorFlow graph file is therefore a concrete candidate reference artifact co-located with the pinned ONNX file, rather than a separately selected TensorFlow Model Zoo download.

The Hugging Face file metadata for the TensorFlow graph identifies:
- File: `ssd-mobilenet-v2.pb`
- Size: 67,386,385 bytes
- SHA256: `b3a28ca449b75b5aefe763f406150ad65f3e1bc0398fcc4b083f0da526cd5d0b`
- File upload commit: `0d69759efbc58eec472934ce08f62d15de947e80`

The ONNX artifact metadata identifies SHA256 `f0d9458cd8ae5c0cde8eadf22b75c6451059f4d9dd46df266e996dcbec417a8f`. Both files are present in the pinned repository snapshot, but co-location alone does not prove that the TensorFlow graph is the exact graph used to produce the ONNX artifact or that preprocessing and exported tensor semantics are identical.

Next verification step: download the pinned `.pb` artifact, verify its SHA256 before parsing, inspect its GraphDef input/output and intermediate tensor names, and compare numerically against the ONNX model on identical, explicitly documented input tensors. Do not enable SSD decoding unless the compared tensors, preprocessing, layout, and numerical tolerances are established. TensorFlow runtime installation is intentionally not prescribed until the graph format and execution requirements have been inspected.
