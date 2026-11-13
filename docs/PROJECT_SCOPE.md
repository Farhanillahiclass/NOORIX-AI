# Project Scope

## Academic title
Offline AI-Based Multimodal Accessibility Assistant for Visually Impaired Users.

## Goal
Explore whether a laptop-based camera and audio interface can provide useful, explainable descriptions of selected visual information.

## Planned prototype capabilities
1. Estimate Pakistani banknote denomination.
2. Detect selected everyday objects and describe their approximate left/centre/right position in the camera image.
3. Read clear printed English text (bonus capability).
4. Provide spoken feedback; Urdu audio is planned for currency and object outputs.

## Hardware
A laptop, camera/webcam, and speaker or headphones are sufficient for the documented laptop prototype. No separate distance or navigation sensors are specified.

## Explicit exclusions
- Banknote authenticity or counterfeit detection.
- Distance estimation, obstacle avoidance, route planning, or autonomous navigation.
- Urdu OCR in the initial scope.
- Claims of production readiness or measured accuracy before testing.

## Acceptance gates
- Each model is documented with its source, license, version, intended use, and known limitations.
- Unit tests pass.
- Live camera tests are performed on the target laptop.
- Recognition results are measured on a documented test set.
- Offline operation is verified with network access disabled after all required assets are installed.
- Accessibility and safety wording is reviewed before public demonstration.
