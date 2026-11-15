# Camera and printed-text quality improvements

The camera adapter requests 1280x720 capture and autofocus where the webcam
driver exposes those controls. Drivers may ignore requested properties; use
the actual frame dimensions as the source of truth. A higher requested
resolution does not guarantee sharper focus or better optics.

The printed OCR adapter accepts PIL images and OpenCV BGR/BGRA frames. It
converts camera channel order correctly, looks for a large four-corner page
region, and attempts perspective correction when a reliable page boundary is
found. It also applies grayscale/contrast/sharpening, upscales text, and tries
an Otsu-thresholded variant. If page detection is uncertain, it falls back to
the full frame. English and Urdu still require their corresponding Tesseract
language data.

The dashboard's Read printed page action now includes frame-quality warnings
for low light, low contrast, blur, or insufficient resolution. The camera
status line reports the actual dimensions when the driver provides them.

For best results, fill the frame with one page, keep the page flat, avoid
covering text with fingers, reduce glare, and allow autofocus to settle.
This implementation is for printed text; handwriting, document boundary
detection, guaranteed transcription, and automatic navigation are not
claimed. Object positions remain relative to the image only.
