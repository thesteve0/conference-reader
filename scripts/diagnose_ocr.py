#!/usr/bin/env python3
"""Find images that return empty OCR results."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conference_reader.config import apply_rocm_stability_settings

OUTPUT_FILE = Path("ocr_diagnosis.json")
IMAGE_DIR = Path("/data/neurips/posters")
TIMEOUT = 30.0
SCALE = 0.75
MAX_EMPTY = 2


def save_results(results: list):
    """Save results to JSON file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def test_image(image_path: str) -> dict:
    """Test one image, return result dict."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(images_scale=SCALE, document_timeout=TIMEOUT)
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=opts),
        }
    )

    start = time.time()
    try:
        result = converter.convert(image_path)
        text = result.document.export_to_markdown()
        return {
            "status": "ok",
            "time": round(time.time() - start, 2),
            "chars": len(text),
        }
    except Exception as e:
        return {
            "status": "error",
            "time": round(time.time() - start, 2),
            "error": str(e)[:80],
        }


def main():
    apply_rocm_stability_settings()

    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.JPEG"))
    print(f"Found {len(images)} images")

    results = []
    empty_count = 0

    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name}...", end=" ", flush=True)

        r = test_image(str(img))
        r["filename"] = img.name
        results.append(r)

        # Save after each image
        save_results(results)

        if r["status"] == "ok" and r["chars"] > 50:
            print(f"OK {r['chars']} chars ({r['time']}s)")
        else:
            print(f"EMPTY/FAIL {r.get('chars', 0)} chars ({r['time']}s)")
            empty_count += 1
            if empty_count >= MAX_EMPTY:
                print(f"\nFound {MAX_EMPTY} empty results. Stopping.")
                break

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("\nEmpty/failing images:")
    for r in results:
        if r["status"] != "ok" or r.get("chars", 0) <= 50:
            print(f"  {r['filename']}: {r}")


if __name__ == "__main__":
    main()
