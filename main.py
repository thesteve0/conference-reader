"""
Conference Reader - Main Application Entry Point

This application processes conference poster images through a pipeline:
1. Discover images in a directory
2. Classify images (filter posters vs QR codes) using VLM
3. Extract text from poster images using Docling
4. Summarize extracted text using SmolLM3
5. Export results to CSV

Usage:
    python main.py
    python main.py --directory /path/to/images
    python main.py -d /path/to/images -o results.csv

Environment:
    - Runs inside devcontainer with AMD GPU (ROCm) access
    - Requires .venv to be activated
"""

import argparse
import sys
from pathlib import Path

from conference_reader.config import apply_rocm_stability_settings
from conference_reader.image_loader import ImageLoader
from conference_reader.classifier import ImageClassifier
from conference_reader.extraction import (
    DocumentExtractor,
    ProcessedDocument,
)
from conference_reader.summarization import TextSummarizer
from conference_reader.output import CSVExporter


DEFAULT_IMAGE_DIRECTORY = "/data/neurips/posters"
DEFAULT_OUTPUT_FILE = "posters.csv"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Process conference poster images into a summarized CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=DEFAULT_IMAGE_DIRECTORY,
        help="Directory containing poster images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV filename",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=0.75,
        help="Scale factor for image resolution (default: 0.75)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum seconds per image (default: 120)",
    )
    return parser.parse_args()


def print_results(documents: list[ProcessedDocument]) -> None:
    """Print extraction results to console.

    Args:
        documents: List of ProcessedDocument instances to display
    """
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80 + "\n")

    successful = [doc for doc in documents if doc.success]
    failed = [doc for doc in documents if not doc.success]

    print(f"Total documents: {len(documents)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}\n")

    for doc in successful:
        print(f"\n{'─' * 80}")
        print(f"FILE: {doc.filename}")
        if doc.title:
            print(f"TITLE: {doc.title}")
        if doc.summary:
            print(f"SUMMARY: {doc.summary}")
        print(f"{'─' * 80}")

    if failed:
        print(f"\n{'─' * 80}")
        print("FAILED:")
        print(f"{'─' * 80}")
        for doc in failed:
            print(f"  - {doc.filename}: {doc.error_message}")
        print()


def main():
    """Main application entry point."""
    args = parse_arguments()

    # Apply ROCm stability settings before any GPU operations
    apply_rocm_stability_settings()

    try:
        # Step 1: Discover images
        print(f"Scanning directory: {args.directory}")
        loader = ImageLoader(directory=args.directory)
        image_paths = loader.get_image_paths()
        print(f"Found {len(image_paths)} image(s)\n")

        if not image_paths:
            print("No images found. Exiting.")
            return

        # Step 2: Classify images (filter posters vs QR codes)
        print("Classifying images (poster vs QR code)...")
        classifier = ImageClassifier()
        poster_paths = classifier.filter_posters(
            [Path(p) for p in image_paths],
            verbose=args.verbose,
        )
        classifier.unload()  # Free GPU memory before loading next model
        print(f"Found {len(poster_paths)} poster(s)\n")

        if not poster_paths:
            print("No posters found after classification. Exiting.")
            return

        # Step 3: Extract text from poster images
        print("Extracting text from posters...")
        extractor = DocumentExtractor(
            images_scale=args.images_scale,
            document_timeout=args.timeout,
        )
        documents = extractor.extract_batch(
            [str(p) for p in poster_paths],
            verbose=args.verbose,
        )
        successful_docs = [doc for doc in documents if doc.success]
        print(f"Extracted text from {len(successful_docs)} poster(s)\n")

        # Step 4: Summarize extracted text
        print("Generating summaries...")
        summarizer = TextSummarizer()
        documents = summarizer.summarize_batch(documents)
        print("Summaries generated\n")

        # Step 5: Export to CSV
        print("Exporting to CSV...")
        exporter = CSVExporter()
        output_path = exporter.export(documents, args.output)
        print(f"Results exported to: {output_path}\n")

        # Display results
        print_results(documents)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
