# OCR Failure Investigation

## Problem Summary

When processing conference poster images with Docling/RapidOCR, some images cause:
1. **Empty OCR results** - RapidOCR returns "text detection result is empty"
2. **Stuck threads** - OCR thread hangs indefinitely, not using GPU
3. **Resource exhaustion** - After a stuck thread, subsequent images also fail

## Observed Behavior

From processing 150 poster images:
- Images 1-6: Process normally (3-12 seconds each)
- Image 7 (IMG_1171.JPEG): Empty OCR result, takes 115 seconds
- Image 8 (IMG_1172.JPEG): Empty OCR result, 5 seconds
- Images 9-14: Normal processing
- Image 15 (IMG_1209.JPEG): Timeout at 120 seconds, OCR thread stuck
- After stuck thread: GPU stops being used, subsequent images timeout

## Key Log Messages

```
[WARNING] RapidOCR main.py:125: The text detection result is empty
RapidOCR returned empty result!

Stage ocr thread did not terminate within 15s. Thread is likely stuck in a blocking call and will be abandoned (resources may leak).
```

## Investigation Tasks

### 1. Identify All Failing Images

Run this script to test each image and record results:

```python
# scripts/diagnose_ocr.py
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def test_single_image(image_path: str, timeout: float = 60.0):
    """Test a single image and return diagnostic info."""
    pipeline_options = PdfPipelineOptions(
        images_scale=0.75,
        document_timeout=timeout,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    start = time.time()
    try:
        result = converter.convert(image_path)
        elapsed = time.time() - start
        text = result.document.export_to_markdown()
        return {
            "status": "success",
            "time": elapsed,
            "text_length": len(text),
            "text_preview": text[:200] if text else "(empty)",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "error",
            "time": elapsed,
            "error": str(e),
        }

if __name__ == "__main__":
    image_dir = Path("/data/neurips/posters")
    results = []

    for img_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPEG")):
        print(f"Testing: {img_path.name}")
        result = test_single_image(str(img_path), timeout=30.0)
        result["filename"] = img_path.name
        results.append(result)
        print(f"  -> {result['status']}: {result.get('text_length', 0)} chars in {result['time']:.1f}s")

        # Stop after finding 5 failures to avoid getting stuck
        failures = [r for r in results if r["status"] != "success" or r.get("text_length", 0) < 100]
        if len(failures) >= 5:
            print("\nFound 5 failing images, stopping early.")
            break

    print("\n=== FAILING IMAGES ===")
    for r in results:
        if r["status"] != "success" or r.get("text_length", 0) < 100:
            print(f"{r['filename']}: {r}")
```

### 2. Examine Failing Images

Questions to answer:
- What's different about failing images? (resolution, format, content type)
- Are they rotated, blurry, or have unusual aspect ratios?
- Do they have non-Latin text that RapidOCR can't handle?
- Are they mostly graphics with minimal text?

Commands to check image properties:
```bash
# Get image info
identify /data/neurips/posters/IMG_1171.JPEG
identify /data/neurips/posters/IMG_1209.JPEG

# Compare with a working image
identify /data/neurips/posters/014F7433-6DEE-4C1B-BB38-53BE406072A5.jpg
```

### 3. Test Different OCR Configurations

Try these variations:
- Different `images_scale` values (0.5, 1.0, 1.5)
- Different OCR engines (tesseract vs rapidocr)
- Image preprocessing (resize, convert to PNG)

## Environment Info

- GPU: AMD ROCm (Framework 13 laptop)
- Python: 3.12
- RapidOCR with torch backend
- Docling for document conversion

## Key Code Locations

- `src/conference_reader/extraction/document_extractor.py` - DocumentExtractor class
- `src/conference_reader/config/rocm_config.py` - GPU settings

## Hypothesis

The most likely causes:
1. **Image content** - Images are graphics-heavy with no detectable text regions
2. **Image format/encoding** - Something about JPEG encoding confuses RapidOCR
3. **Resolution mismatch** - Images too large or too small for the OCR model
4. **GPU memory** - After a failure, GPU memory is corrupted/exhausted

## Next Steps

1. Run the diagnostic script to identify all failing images
2. Visually inspect the failing images to find common patterns
3. Test if the same images fail with tesseract OCR
4. Consider pre-processing (resize, format conversion) for problematic images
