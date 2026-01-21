"""
Deep dive into Docling's timings and confidence score structure.
"""

from docling.document_converter import DocumentConverter
from pathlib import Path
import json


def explore_timings_and_confidence(image_path: str):
    """Detailed investigation of timings and confidence attributes."""
    print(f"Processing: {image_path}\n")
    print("="*80)

    converter = DocumentConverter()
    result = converter.convert(image_path)

    # 1. TIMINGS - Deep investigation
    print("\n1. TIMINGS INVESTIGATION:")
    print("-"*80)
    print(f"  Timings type: {type(result.timings)}")
    print(f"  Timings value: {result.timings}")
    print(f"  Timings dict content: {dict(result.timings)}")

    if result.timings:
        print(f"  Timings keys: {list(result.timings.keys())}")
        for key, value in result.timings.items():
            print(f"    {key}: {value} (type: {type(value)})")
    else:
        print("  Timings is empty or None")

    # Check for other timing-related attributes
    if hasattr(result, 'timestamp'):
        print(f"  Timestamp: {result.timestamp}")

    # 2. CONFIDENCE SCORES - Detailed breakdown
    print("\n2. CONFIDENCE SCORES DETAILED BREAKDOWN:")
    print("-"*80)

    conf = result.confidence
    print(f"  Confidence object type: {type(conf)}")

    # Individual scores
    print(f"\n  Individual Scores:")
    print(f"    parse_score:  {conf.parse_score} (digital text quality, 10th percentile)")
    print(f"    layout_score: {conf.layout_score} (document element recognition quality)")
    print(f"    table_score:  {conf.table_score} (table extraction - not yet implemented)")
    print(f"    ocr_score:    {conf.ocr_score} (OCR content quality)")

    # Summary scores
    print(f"\n  Summary Scores:")
    print(f"    mean_score: {conf.mean_score} (average of component scores)")
    print(f"    low_score:  {conf.low_score} (5th percentile - highlights worst areas)")

    # Quality grades
    print(f"\n  Quality Grades (categorical):")
    print(f"    mean_grade: {conf.mean_grade} (overall quality)")
    print(f"    low_grade:  {conf.low_grade} (worst-performing areas)")

    # Per-page confidence
    print(f"\n  Per-Page Confidence:")
    print(f"    Pages in confidence: {list(conf.pages.keys())}")
    for page_idx, page_conf in conf.pages.items():
        print(f"\n    Page {page_idx}:")
        print(f"      parse_score:  {page_conf.parse_score}")
        print(f"      layout_score: {page_conf.layout_score}")
        print(f"      table_score:  {page_conf.table_score}")
        print(f"      ocr_score:    {page_conf.ocr_score}")
        print(f"      mean_score:   {page_conf.mean_score}")
        print(f"      low_score:    {page_conf.low_score}")
        print(f"      mean_grade:   {page_conf.mean_grade}")
        print(f"      low_grade:    {page_conf.low_grade}")

    # 3. EXPORT FORMAT COMPARISON
    print("\n3. EXPORT FORMAT COMPARISON FOR SMOLLM3:")
    print("-"*80)

    doc = result.document

    # Markdown export
    md_output = doc.export_to_markdown()
    print(f"\n  MARKDOWN (export_to_markdown):")
    print(f"    Length: {len(md_output)} characters")
    print(f"    First 300 chars:\n{md_output[:300]}")
    print(f"    ...\n")

    # Text export
    text_output = doc.export_to_text()
    print(f"  TEXT (export_to_text):")
    print(f"    Length: {len(text_output)} characters")
    print(f"    First 300 chars:\n{text_output[:300]}")
    print(f"    ...\n")

    # Dict export structure
    dict_output = doc.export_to_dict()
    print(f"  DICT (export_to_dict):")
    print(f"    Type: {type(dict_output)}")
    print(f"    Top-level keys: {list(dict_output.keys())}")
    print(f"    Body content preview:")
    if 'body' in dict_output:
        print(f"      Body type: {type(dict_output['body'])}")
        print(f"      Body structure: {str(dict_output['body'])[:300]}")

    # HTML export
    html_output = doc.export_to_html()
    print(f"\n  HTML (export_to_html):")
    print(f"    Length: {len(html_output)} characters")
    print(f"    First 300 chars:\n{html_output[:300]}")

    # 4. RECOMMENDATIONS
    print("\n4. RECOMMENDATIONS FOR SMOLLM3:")
    print("-"*80)
    print("""
    MARKDOWN (export_to_markdown):
      ✓ Preserves document structure (headings, lists, etc.)
      ✓ Good balance of formatting and readability
      ✓ Most LLMs are trained on markdown
      → RECOMMENDED for SmolLM3

    TEXT (export_to_text):
      ✓ Plain text, no formatting
      ✓ Smallest size
      ✗ Loses document structure
      → Use if token count is critical

    DICT (export_to_dict):
      ✓ Full structured data
      ✓ Access to specific elements (tables, pictures, etc.)
      ✗ Complex structure, needs processing
      → Use if you need programmatic access to elements

    HTML (export_to_html):
      ✓ Rich formatting
      ✗ Verbose, larger token count
      ✗ May confuse some LLMs
      → Not recommended for SmolLM3
    """)

    print("\n" + "="*80)


if __name__ == "__main__":
    test_image = "/data/neurips/poster_test/good-test.jpg"

    if not Path(test_image).exists():
        print(f"Error: Test image not found at {test_image}")
    else:
        explore_timings_and_confidence(test_image)
