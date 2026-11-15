# Camera and printed-text quality improvements

The camera adapter requests 1280x720 capture and autofocus where the webcam
driver exposes those controls. Drivers may ignore requested properties; use
the actual frame dimensions as the source of truth. A higher requested
resolution does not guarantee sharper focus or better optics.

The printed OCR adapter accepts PIL images and OpenCV BGR/BGRA frames. It
converts camera channel order correctly, applies grayscale/contrast/sharpening,
upscales text, and tries an Otsu-thresholded variant. It selects the strongest
text candidate from the attempts. English and Urdu still require their
corresponding Tesseract language data.

For best results, fill the frame with one page, keep the page flat, avoid
covering text with fingers, reduce glare, and allow autofocus to settle.
This implementation is for printed text; handwriting, document boundary
detection, guaranteed transcription, and automatic navigation are not
claimed. Object positions remain relative to the image only.
