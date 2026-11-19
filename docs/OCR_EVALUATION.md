# NOORIX OCR Evaluation Protocol

This protocol turns a human-verified image/text manifest into a repeatable OCR report. It is an evaluation aid, not a safety certification.

## Dataset requirements

- Use images you have permission to process and retain.
- Obtain informed consent where images contain personal or sensitive information.
- Keep the test set held out from tuning and model/preprocessing selection.
- Include separate, adequately sized English and Urdu subsets.
- Record document type, capture device, lighting, font/print size, blur, perspective, contrast and other relevant conditions in dataset notes.
- Have reference transcriptions checked by a fluent human reader. Keep references immutable after evaluation begins.
- Do not upload private documents or transcripts to public repositories.

## Manifest

Create a UTF-8 CSV with these required columns: image,reference,language.

Example rows:

    image,reference,language
    images/english_001.png,The exact printed reference.,eng
    images/urdu_001.png,اصل مطبوعہ عبارت۔,urd

Image paths are relative to the CSV file. Supported language values are eng, urd, and eng+urd. Keep English and Urdu results separate; do not use a single aggregate score to hide language-specific failures.

## Run

Install optional OCR dependencies and ensure the Tesseract executable plus matching language data are installed:

    python -m pip install -e ".[ocr]"
    python scripts/evaluate_ocr.py datasets/ocr/manifest.csv --output reports/ocr-evaluation.json

The JSON report records per-sample hypotheses and aggregate/per-language character error rate (CER) and word error rate (WER). CER ignores whitespace but preserves punctuation and case. WER uses whitespace-delimited tokens. Review sample-level errors; aggregate scores alone can conceal critical misses.

## Reporting rules

Every published result must state sample count, language, dataset provenance, split, capture conditions, OCR/Tesseract versions, and known exclusions. Define acceptance thresholds before running the held-out evaluation. Repeat runs after any model, language-data, preprocessing or dependency change.

Do not infer field safety, navigation reliability, handwriting support, or production readiness from OCR metrics. A qualified accessibility review and supervised user evaluation remain separate release gates.
