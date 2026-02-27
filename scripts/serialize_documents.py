"""
Serialize ProcessedDocuments for Experimentation

This script extracts text from test images using EasyOCR and serializes
the resulting ProcessedDocuments to a file. This allows experimentation
scripts to load pre-processed documents instead of running OCR repeatedly,
speeding up the SmolLM3 experimentation workflow.

Usage:
    python serialize_documents.py                    # Use default directory
    python serialize_documents.py -d /path/to/images # Custom directory
"""

import argparse
import pickle
import sys
from pathlib import Path

from conference_reader.image_loader import ImageLoader
from conference_reader.extraction import DocumentExtractor


DEFAULT_IMAGE_DIRECTORY = "/data/neurips/poster_test"
OUTPUT_FILE = "output/serialized_document_extractions.pkl"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and serialize ProcessedDocuments from images"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=DEFAULT_IMAGE_DIRECTORY,
        help=f"Directory containing poster images (default: {DEFAULT_IMAGE_DIRECTORY})"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=OUTPUT_FILE,
        help=f"Output file path (default: {OUTPUT_FILE})"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Step 1: Discover images
        print(f"Scanning directory: {args.directory}")
        loader = ImageLoader(directory=args.directory)
        image_paths = loader.get_image_paths()
        print(f"Found {len(image_paths)} image(s)\n")

        if not image_paths:
            print("No images found. Nothing to serialize.")
            sys.exit(0)

        # Step 2: Extract text using EasyOCR
        print("Extracting text from images (this may take a minute)...")
        extractor = DocumentExtractor()
        documents = extractor.extract_batch(image_paths)

        # Step 3: Report extraction results
        successful = [doc for doc in documents if doc.success]
        failed = [doc for doc in documents if not doc.success]
        print(f"\nExtraction complete:")
        print(f"  - Successful: {len(successful)}")
        print(f"  - Failed: {len(failed)}")

        if failed:
            print("\nFailed extractions:")
            for doc in failed:
                print(f"  - {doc.filename}: {doc.error_message}")

        # Step 4: Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 5: Serialize to file
        print(f"\nSerializing {len(documents)} documents to: {args.output}")
        with open(output_path, 'wb') as f:
            pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Verify file was created
        file_size_kb = output_path.stat().st_size / 1024
        print(f"✓ Serialization complete! File size: {file_size_kb:.2f} KB")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
