# OCR Improvement Notes

These changes showed promise for improving OCR quality on conference poster photos but were reverted due to integration issues. Consider re-applying incrementally.

## Tesseract Settings (Low Risk)

In `document_extractor.py`, change `_create_converter()`:

```python
# Before (suboptimal):
ocr_options = TesseractCliOcrOptions(lang=["eng"])
images_scale=0.75  # downscaling hurts OCR

# After (improved):
ocr_options = TesseractCliOcrOptions(
    lang=["eng"],
    psm=11,  # Sparse text mode - better for complex poster layouts
    force_full_page_ocr=True,  # OCR entire page, don't skip "image" regions
)

pipeline_options = PdfPipelineOptions(
    images_scale=1.5,  # Higher resolution improves Tesseract accuracy
    ...
)
```

Also update `__init__` default: `images_scale: float = 1.5`

And `main.py` argument default: `default=1.5`

## Deskew Preprocessing (Higher Risk)

OpenCV-based deskew showed improvement on angled photos but caused integration issues.

Key approach:
1. Use Hough line detection to find text line angles
2. Calculate median angle (robust to outliers)
3. Rotate image to correct skew
4. Pass corrected image to OCR

Dependencies needed: `opencv-python-headless`

The deskew logic worked in isolation but the integration with the batch processing pipeline caused GPU issues.

## Test Results Before Revert

With Tesseract settings + deskew enabled:
- IMG_1222: 0 → 1,067 chars extracted
- IMG_1230: 91 → 972 chars extracted
- IMG_1242: 0 → 4,621 chars extracted

## Recommendation

1. First apply only the Tesseract settings changes (psm=11, force_full_page_ocr, images_scale=1.5)
2. Test full pipeline
3. If working, consider adding deskew as a separate step
