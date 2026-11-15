# NOORIX-AI Kaggle OCR Training Notebooks

Prepared by **Muhammad Farhan**.

These two notebooks are starter fine-tuning pipelines for line-level OCR:

- `printed_ocr_training_muhammad_farhan.ipynb` — printed books/documents.
- `handwritten_ocr_training_muhammad_farhan.ipynb` — handwritten text.

## Important language limitation

The selected TrOCR checkpoints (`microsoft/trocr-base-printed` and `microsoft/trocr-base-handwritten`) are English-oriented pretrained checkpoints. The notebooks can tokenize Unicode text and can be used for experimental fine-tuning with Urdu transcriptions, but that does **not** mean they provide reliable Urdu OCR out of the box. Urdu needs its own correctly labelled train/validation/test data and separate evaluation. Do not mix English and Urdu blindly or claim Urdu support without measured results.

## Required Kaggle dataset format

Attach your image dataset using Kaggle **Add Input**. The dataset must include a CSV manifest with one row per text-line image:

```csv
image_path,text
lines/0001.png,This is the exact printed line.
lines/0002.png,This is another exact line.
```

- `image_path`: absolute Kaggle path or path relative to the attached dataset.
- `text`: exact transcription of that image, preserving spelling and punctuation.
- Each image should contain one line or a short text crop, not an entire page.
- For Urdu, store the exact Unicode Urdu transcription in `text`.
- A folder of images without transcriptions is **not** enough to train OCR.

## Run on Kaggle

1. Open the notebook from this repository and choose **Download** or **Open in Kaggle** if available.
2. Create a new Kaggle notebook and import the downloaded `.ipynb`.
3. Use **Add Input** to attach your labelled dataset.
4. Enable GPU if Kaggle makes one available.
5. Run the first cells and inspect the discovered CSV files.
6. In the manifest cell, set `ANNOTATION_CSV` to the correct CSV path. Set `IMAGE_COL` and `TEXT_COL` only if auto-detection does not find your column names.
7. Run the notebook. It reports CER/WER and saves a model/processor folder plus a ZIP in `/kaggle/working/`.

Lower CER/WER is better. Inspect predictions manually and retain a separate real-world test set before integrating a model into NOORIX-AI. The saved model is not yet an ONNX deployment package; integration requires matching preprocessing, tokenizer, model architecture and runtime.

## Compute

Start with the default 3 epochs and small batch size. If GPU memory runs out, reduce batch size to 2. Training may take significant time. Do not enable internet or download checkpoints unless permitted by your Kaggle account and dataset rules.

## License and data

Check each dataset and pretrained model's license and terms before redistribution or commercial use. Do not upload private documents or personal handwriting without permission.
