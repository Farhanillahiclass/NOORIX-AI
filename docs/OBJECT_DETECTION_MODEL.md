# YOLOv8 ONNX object detection

NOORIX includes an optional CPU inference adapter for a standard YOLOv8 object-detection ONNX export trained on the 80 COCO classes.

## Model file

The ONNX weights are intentionally not committed to Git. Put a compatible model at:

`models/object_detection/yolov8n.onnx`

The adapter expects:
- one NCHW input: `[1, 3, 640, 640]` by default;
- exactly one raw detection output shaped `[1, 84, N]` or `[1, N, 84]`, with a statically visible 84-value feature axis when model metadata provides fixed dimensions;
- float32 output, 4 box values (center x, center y, width, height) and 80 class scores per candidate;
- the canonical COCO-80 labels in their original class-index order (the runtime rejects reordered, duplicate, blank, or custom labels);
- a standard detection export, not segmentation, pose, custom-class, or end-to-end/NMS-included output.

Use the exact YOLOv8n COCO-80 ONNX model that was already exported and tested in the NOORIX model workspace. Do not rename a different model and assume compatibility. The runtime validates input and output metadata before the first inference, then validates the actual output tensor shape and caps candidate count at 10,000 to bound class-aware NMS work. It also checks that the supplied label map exactly matches the canonical COCO-80 class order, preventing silent class-name mismatches.

## Install and configure

Install the optional CPU inference dependencies from the repository root:

```bash
python -m pip install -e ".[camera,onnx,dev]"
```

Copy the compatible model into `models/object_detection/yolov8n.onnx`, then copy `config.example.json` to `config.json`. The example config selects the `yolov8_onnx` backend and the ordered `yolov8_coco80_labels.json` file.

The model is loaded with ONNX Runtime's CPU execution provider. OpenCV is used for image resizing and letterbox preprocessing. No GPU is required.

## Dataset for real-image checks

Recommended first evaluation source: **COCO 2017 validation images** and their official validation annotations.

- Images: https://images.cocodataset.org/zips/val2017.zip
- Annotations archive: https://images.cocodataset.org/annotations/annotations_trainval2017.zip
- Terms: https://cocodataset.org/dataset/termsofuse.htm

Download and extract the validation images and annotations locally, outside Git. The COCO Consortium licenses its annotations under CC BY 4.0, but it does not own the images; image use is governed by the original Flickr terms. Review the official terms and image-level rights for the intended use before downloading, sharing, or redistributing any images.

The validation set is suitable for a real-world inference smoke test and, with a separate COCO evaluation implementation, benchmark evaluation. Running inference on a few images is not an accuracy metric. Do not report mAP, precision, recall, or model superiority unless the official annotations are evaluated with a documented metric implementation and exact split.

## Current limitations

- The runtime supports the standard raw YOLOv8 COCO-80 detection head only.
- It performs class-aware non-maximum suppression and maps letterboxed boxes back to original image coordinates.
- It is CPU-capable but speed depends on the user's processor and image size.
- The repository does not bundle model weights or COCO images.
- YOLOv8 software and model weights are distributed under AGPL-3.0 by default, with an Enterprise licensing option. Review the full license obligations before redistributing or embedding these weights in a product: https://docs.ultralytics.com/models/yolov8
- This integration does not train or fine-tune the model. It runs inference with an existing pretrained checkpoint.

## CLI / application integration

The application model factory accepts:

```json
{
  "backend": "yolov8_onnx",
  "model_path": "models/object_detection/yolov8n.onnx",
  "labels_path": "models/object_detection/yolov8_coco80_labels.json",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "input_size": 640,
  "max_detections": 100
}
```

The detector returns the same core result fields as the existing object detector: label, class ID, confidence, horizontal position, and pixel bounding box.


## CPU post-processing bounds

Raw YOLO output is limited to 10,000 candidates. Per-class NMS considers at most the 3,000 highest-scoring candidates to bound worst-case quadratic suppression work on CPU-only devices. This may discard lower-ranked proposals in unusually dense scenes; representative held-out recall evaluation is required before treating the cap as suitable for a release.
