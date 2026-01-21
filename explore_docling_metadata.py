"""
Exploration script to inspect Docling result object structure.

This script processes a single test image and prints all available
attributes and metadata from the Docling result object.
"""

from docling.document_converter import DocumentConverter
from pathlib import Path
import json


def explore_result_structure(image_path: str):
    """Inspect and print Docling result object structure."""
    print(f"Processing: {image_path}\n")
    print("="*80)

    converter = DocumentConverter()
    result = converter.convert(image_path)

    # 1. Top-level result object attributes
    print("\n1. TOP-LEVEL RESULT ATTRIBUTES:")
    print("-"*80)
    result_attrs = [attr for attr in dir(result) if not attr.startswith('_')]
    for attr in result_attrs:
        print(f"  - {attr}")

    # 2. Document object attributes
    print("\n2. DOCUMENT OBJECT ATTRIBUTES:")
    print("-"*80)
    doc = result.document
    doc_attrs = [attr for attr in dir(doc) if not attr.startswith('_')]
    for attr in doc_attrs:
        print(f"  - {attr}")

    # 3. Try to access common metadata fields
    print("\n3. ACCESSING METADATA VALUES:")
    print("-"*80)

    # Check for pages
    try:
        if hasattr(doc, 'pages'):
            print(f"  Number of pages: {len(doc.pages)}")
            print(f"  Pages type: {type(doc.pages)}")
            if doc.pages:
                # Pages might be a dict, not a list
                if isinstance(doc.pages, dict):
                    first_key = list(doc.pages.keys())[0]
                    print(f"  First page key: {first_key}")
                    print(f"  First page type: {type(doc.pages[first_key])}")
                else:
                    print(f"  First page type: {type(doc.pages[0])}")
    except Exception as e:
        print(f"  Error accessing pages: {e}")

    # Check for metadata
    try:
        if hasattr(doc, 'metadata'):
            print(f"  Metadata: {doc.metadata}")
    except Exception as e:
        print(f"  Error accessing metadata: {e}")

    # Check for confidence/score
    try:
        if hasattr(result, 'score'):
            print(f"  Confidence score: {result.score}")
        if hasattr(result, 'confidence'):
            print(f"  Confidence: {result.confidence}")
    except Exception as e:
        print(f"  Error accessing confidence: {e}")

    # Check for layout information
    try:
        if hasattr(doc, 'layout'):
            print(f"  Layout info: {type(doc.layout)}")
    except Exception as e:
        print(f"  Error accessing layout: {e}")

    # Check for bounding boxes
    try:
        if hasattr(doc, 'bboxes'):
            print(f"  Bounding boxes: {doc.bboxes}")
    except Exception as e:
        print(f"  Error accessing bboxes: {e}")

    # 4. Inspect the first page if available
    try:
        if hasattr(doc, 'pages') and doc.pages:
            print("\n4. FIRST PAGE ATTRIBUTES:")
            print("-"*80)

            # Handle dict vs list pages
            if isinstance(doc.pages, dict):
                first_key = list(doc.pages.keys())[0]
                page = doc.pages[first_key]
            else:
                page = doc.pages[0]

            page_attrs = [attr for attr in dir(page) if not attr.startswith('_')]
            for attr in page_attrs:
                print(f"  - {attr}")

            # Try to get page-specific info
            if hasattr(page, 'size'):
                print(f"\n  Page size: {page.size}")
            if hasattr(page, 'image'):
                print(f"  Page has image: {hasattr(page, 'image')}")
    except Exception as e:
        print(f"  Error inspecting page: {e}")

    # 5. Export formats available
    print("\n5. EXPORT METHODS:")
    print("-"*80)
    export_methods = [m for m in dir(doc) if 'export' in m.lower() and not m.startswith('_')]
    for method in export_methods:
        print(f"  - {method}")

    # 6. Try different export formats
    print("\n6. SAMPLE EXPORTS:")
    print("-"*80)

    # Markdown (we already use this)
    try:
        md_text = doc.export_to_markdown()
        print(f"  Markdown length: {len(md_text)} characters")
    except Exception as e:
        print(f"  Error exporting to markdown: {e}")

    # Check for dict export (most useful for exploring structure)
    try:
        if hasattr(doc, 'export_to_dict'):
            dict_output = doc.export_to_dict()
            print(f"  Dict export available: {type(dict_output)}")
            if isinstance(dict_output, dict):
                print(f"  Dict top-level keys: {list(dict_output.keys())}")
                # Print first few key-value pairs
                for i, (key, value) in enumerate(dict_output.items()):
                    if i >= 5:  # Only show first 5
                        break
                    print(f"    {key}: {type(value).__name__}")
    except Exception as e:
        print(f"  Error exporting to dict: {e}")

    # 7. Check result-level metadata
    print("\n7. RESULT-LEVEL METADATA:")
    print("-"*80)
    try:
        print(f"  Status: {result.status}")
        print(f"  Timestamp: {result.timestamp}")
        if hasattr(result, 'timings'):
            print(f"  Timings: {result.timings}")
        if hasattr(result, 'errors'):
            print(f"  Errors: {result.errors}")
        if hasattr(result, 'confidence'):
            print(f"  Confidence: {result.confidence}")
    except Exception as e:
        print(f"  Error accessing result metadata: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Use a known good test image
    test_image = "/data/neurips/poster_test/good-test.jpg"

    if not Path(test_image).exists():
        print(f"Error: Test image not found at {test_image}")
        print("Please provide path to a valid test image.")
    else:
        explore_result_structure(test_image)
