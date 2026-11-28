# Project Scope

## Academic title
Offline AI-Based Multimodal Accessibility Assistant for Visually Impaired Users.

## Goal
Explore whether a laptop-based camera and audio interface can provide useful, explainable descriptions of selected visual information.

## Planned prototype capabilities
1. Estimate Pakistani banknote denomination using a compatible locally supplied classifier.
2. Detect selected everyday objects and describe their approximate left/centre/right position in the camera image.
3. Read clear printed English and Urdu text when the matching Tesseract language data is installed (optional capability).
4. Provide spoken feedback; Urdu audio is planned for currency and object outputs.

## Hardware
A laptop, camera/webcam, and speaker or headphones are sufficient for the documented laptop prototype. No separate distance or navigation sensors are specified.

## Explicit exclusions
- Banknote authenticity or counterfeit detection.
- Distance estimation, obstacle avoidance, route planning, or autonomous navigation.
- Handwriting recognition and guaranteed OCR accuracy.
- Claims of production readiness or measured accuracy before testing.

## Acceptance gates
- Each model is documented with its source, license, version, intended use, and known limitations.
- Unit tests pass.
- Live camera tests are performed on the target laptop.
- Recognition results are measured on a documented test set.
- Offline operation is verified with network access disabled after all required assets are installed.
- Accessibility and safety wording is reviewed before public demonstration.


## Scope clarification

Urdu OCR was outside the original prototype scope but has since been added as an optional printed-text language. Its presence in the code does not establish accuracy on representative documents; language-data availability and held-out evaluation remain required. Microphone speech-to-text remains outside the implemented feature set.
