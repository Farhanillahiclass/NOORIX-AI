# NOORIX-AI Kaggle Model Lab
**Owner / prepared by: Muhammad Farhan**

All Kaggle training notebooks live in `notebooks/` on branch `feature/kaggle-ocr-training-notebooks`.

> **Accuracy promise:** No responsible workflow can promise “99% functional” or 99% model accuracy in advance. Real performance depends on camera quality, lighting, dataset quality, language, handwriting variation, hardware, and independent testing. We will measure each feature separately and expose uncertainty instead of pretending every result is correct.

## Master orchestrator

`NOORIX_MASTER_AI_TRAINING_MUHAMMAD_FARHAN.ipynb` runs the specialist notebooks as isolated stages, records stage failures without stopping later stages, and packages produced model archives into per-stage ZIPs plus `NOORIX_AI_ALL_MODELS.zip`. Configure the exact dataset paths in its `CONFIG` cell first. Kaggle datasets must still be attached in the Kaggle Input panel unless a permitted download source is explicitly configured; the notebook does not bypass dataset licenses or Kaggle access controls. Start with one stage and inspect its validation metrics before enabling more.

## 1. Notebook inventory

| Feature | Notebook file | Kaggle notebook title | Dataset to attach | Output ZIP |
|---|---|---|---|---|
| Printed English OCR | `notebooks/ocr/printed_ocr_training_muhammad_farhan.ipynb` | `NOORIX-Printed-OCR-Muhammad-Farhan` | Your own line-image + exact transcription CSV; use printed English line crops | `NOORIX_PRINTED_OCR_Muhammad_Farhan.zip` |
| Handwritten English OCR | `notebooks/ocr/handwritten_ocr_training_muhammad_farhan.ipynb` | `NOORIX-Handwritten-OCR-Muhammad-Farhan` | IAM handwriting line images + transcription manifest, or your own labelled line crops | `NOORIX_HANDWRITTEN_OCR_Muhammad_Farhan.zip` |
| Urdu printed + handwritten OCR | `notebooks/ocr/urdu_printed_handwritten_crnn_ctc_muhammad_farhan.ipynb` | `NOORIX-Urdu-Printed-Handwritten-CRNN-CTC-Muhammad-Farhan` | Printed: UPTI 1.0 / IIIT-Urdu OCR. Handwritten: NUST-UHWR. One dataset per run, converted to `image_path,text` CSV | `NOORIX_Urdu_Line_OCR_Muhammad_Farhan.zip` |
| PKR currency | `notebooks/currency/pkr_mobilenetv2_training_muhammad_farhan.ipynb` | `NOORIX-PKR-Currency-MobileNetV2-Muhammad-Farhan` | `Pakistani Currency Notes Image dataset` (Kaggle slug `muhammadammarabid/pakistanicurrencydataset`); optionally your Afghani/PKR dataset, but do not mix currencies accidentally | `NOORIX_PKR_MobileNetV2_Muhammad_Farhan.zip` |
| Everyday objects + people | `notebooks/detection/yolov8n_objects_person_muhammad_farhan.ipynb` | `NOORIX-Object-Person-YOLOv8n-Muhammad-Farhan` | Pretrained COCO mode: none required. Custom training: a YOLO-format labelled dataset (search Kaggle for `COCO 2017 object detection YOLO format`) | `NOORIX_Object_Person_YOLOv8n_Muhammad_Farhan.zip` |
| Urdu / English speech-to-text | `notebooks/speech/whisper_urdu_english_asr_muhammad_farhan.ipynb` | `NOORIX-ASR-Whisper-Urdu-English-Muhammad-Farhan` | Urdu: Common Voice Urdu. English: Common Voice English. Or your own consented audio + exact transcription CSV | `NOORIX_ASR_Whisper_urdu_Muhammad_Farhan.zip` or `..._english_...` |

The two earlier TrOCR notebooks use English-oriented pretrained checkpoints. They are English OCR starting points, not validated Urdu OCR. Use the dedicated Urdu CRNN/CTC notebook for Urdu line recognition and evaluate it carefully.

## 2. Dataset references and search names

### OCR
- **Printed Urdu:** UPTI 1.0 Urdu Printed Text Image Database (Zenodo DOI: https://doi.org/10.5281/zenodo.21679973) or IIIT-Urdu OCR Dataset (https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-urdu-ocr). Both are line-image OCR resources; check the dataset readme and permitted use.
- **Handwritten Urdu:** NUST-UHWR (https://doi.org/10.5281/zenodo.21674269). It includes line images and Unicode Urdu transcriptions. Convert its official train/validation manifests to the notebook's `image_path,text` CSV.
- **Handwritten English:** Kaggle search phrase: `IAM Handwriting Database`. Confirm access, annotation format, and license before use.
- **Printed/handwritten custom data:** each image must have the exact text transcription. Folder names alone are not OCR labels.

### Currency
- Kaggle dataset: https://www.kaggle.com/datasets/muhammadammarabid/pakistanicurrencydataset
- Alternate Pakistani currency dataset: https://www.kaggle.com/datasets/psychocoder/currency-detection
- Use the dataset already attached to the successful PKR notebook; do not download duplicates. Exclude generic `Back` / `Front` classes if they do not identify a denomination. Classification is not counterfeit detection or banknote authentication.

### Object / person detection
- COCO 2017 includes 80 everyday object classes, including `person`. Official Ultralytics dataset guide: https://docs.ultralytics.com/datasets/detect/coco
- For custom training, dataset must contain a valid `data.yaml`, images and YOLO normalized bounding-box label files. A plain folder of images cannot train object detection.

### Speech-to-text
- Kaggle search phrases: `Mozilla Common Voice Urdu` and `Common Voice English`.
- Urdu Common Voice dataset overview: https://huggingface.co/datasets/azeem-ahmed/Common_Voice_Corpus_22_0_Urdu
- Use only audio for which the dataset terms allow your intended use. Do not attempt to identify speakers. Keep exact transcriptions.

## 3. Kaggle workflow for every notebook

1. Open the notebook file in GitHub.
2. Use **Download raw file** (or open the notebook and download it) to get the `.ipynb`.
3. In Kaggle choose **Create → New Notebook → File → Import Notebook** and upload the downloaded file.
4. Rename the Kaggle notebook to the exact title shown in the table.
5. In the notebook's right-side **Input** panel choose **Add Input** and attach the dataset listed above.
6. Select GPU when available. Free GPU quotas and availability are not guaranteed.
7. Run the first setup / dataset discovery cells. Set the requested path variables (for example `CSV_PATH`, `ANNOTATION_CSV`, or `DATA_DIR`) to the real paths printed by Kaggle.
8. Run all cells. Review validation metrics and sample predictions; do not rely on training accuracy alone.
9. Download the single output ZIP named in the table from Kaggle's Output panel. Keep any paired ONNX external-data file inside the ZIP with its ONNX file.

## 4. What each ZIP should contain

- Classifiers: model weights, ordered labels, preprocessing metadata.
- YOLO: `.pt`, `.onnx`, labels and metadata. YOLO output is **not** compatible with the current SSD output adapter; a YOLO-specific decoder is required before NOORIX integration.
- TrOCR: model folder + processor/tokenizer files + metadata.
- Urdu CRNN/CTC: `.pth`, Unicode vocabulary and metadata.
- Whisper: model folder + processor/tokenizer + metadata. This may be large and slow on a CPU-only PC.

A successful ZIP download is not proof of model quality. Each artifact still needs checksum, load, preprocessing, output-shape, held-out accuracy/CER/WER and CPU-latency checks.

## 5. Speech output (TTS) and voice activity detection (VAD)

NOORIX currently has an operating-system TTS path through `pyttsx3`; this uses installed Windows voices rather than a model trained in Kaggle. Check available voices on the PC. Urdu speech output is not guaranteed unless a suitable Urdu voice is installed and tested. TTS model packaging is a separate task; do not confuse ASR (speech-to-text) with TTS (text-to-speech).

VAD (voice activity detection) can be added as a lightweight pretrained component later. It is not a substitute for ASR and does not need a custom dataset for the first prototype.

## 6. Automatic local ZIP detection (Windows)

1. Download Kaggle model output ZIPs and dataset ZIPs normally into `%USERPROFILE%\\Downloads`.
2. Run `scripts\\import_kaggle_models.bat` from the repository root (double-click also works).
3. The script checks ZIP files modified in the **last 2 hours**. You can change the window with `--since-hours 4` when running the Python script directly.
4. ZIP names beginning `NOORIX_` or `NOORIX-` are treated as model packages; recognized model/config files are extracted into `models/imported/<zip-name>/`.
5. Other recent ZIPs are copied intact into `data/datasets/incoming/` as dataset archives. They are not unpacked into the model folder and are not deleted.
6. ZIPs manually placed in `models/incoming/` are also imported. Original ZIPs are preserved; review `import_manifest.json` after import.

The importer does **not** connect models to app code automatically. Model-specific adapters, exact labels, preprocessing and validation must be confirmed before activation. Dataset ZIPs are collected, not automatically attached to Kaggle or unpacked; large archives should be extracted only after checking free disk space.

## 7. Recommended training order

1. Urdu CRNN/CTC OCR on NUST-UHWR (handwriting) and UPTI/IIIT (printed), as separate runs.
2. English printed and handwritten OCR.
3. PKR MobileNetV2 reproducible run.
4. YOLOv8n pretrained COCO export; custom training only if current COCO classes are insufficient.
5. Whisper Urdu ASR, then English ASR.
6. Test every model locally on the target CPU before integrating it into the live camera/voice pipeline.
