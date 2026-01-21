"""
Conference Reader - Main Application Entry Point

This is the main driver program for the conference-reader project, a ROCm-based
data science application built from the datascience-template-ROCm template.

This application will provide functionality to:
1. Read a directory of images of conference posters (and maybe slides)
2. Extract text and structure from these images
3. Send the text to a language model for summarization
4. Output a list of summaries along with links to the original images

This should enable Red Hat employees to quickly see which posters or slides they want to open and read.

Usage:
    python main.py                                    # Use default directory
    python main.py --directory /path/to/images        # Custom directory
    python main.py -d /path/to/images                 # Short form

Environment:
    - Designed to run inside the devcontainer with AMD GPU (ROCm) access
    - Requires .venv to be activated (should happen automatically in devcontainer)
    - GPU verification: Use 'amd-smi' or 'rocm-smi' to check GPU availability

Dependencies:
    - See pyproject.toml for package requirements
    - ROCm-provided packages (torch, numpy, etc.) come from container base image
    - Additional packages installed via uv into .venv
"""

import argparse
import sys

# Note: The package name uses hyphens in directory but underscores in Python imports
from conference_reader.image_loader import ImageLoader
from conference_reader.extraction import (
    DocumentExtractor,
    ProcessedDocument,
    ValidImageConfig,
)
from conference_reader.summarization import TextSummarizer


DEFAULT_IMAGE_DIRECTORY = "/data/neurips/invalid_poster_images"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract text from conference poster images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=DEFAULT_IMAGE_DIRECTORY,
        help=f"Directory containing poster images (default: {DEFAULT_IMAGE_DIRECTORY})",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=800,
        help="Minimum text length for valid poster (default: 800)",
    )
    parser.add_argument(
        "--require-heading",
        action="store_true",
        default=True,
        help="Require markdown heading for valid poster (default: True)",
    )
    parser.add_argument(
        "--min-heading-count",
        type=int,
        default=3,
        help="Minimum number of headings for valid poster (default: 3)",
    )
    return parser.parse_args()


def print_results(documents: list[ProcessedDocument]) -> None:
    """Print extraction results to console.

    Args:
        documents: List of ProcessedDocument instances to display
    """
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80 + "\n")

    successful = [doc for doc in documents if doc.success]
    failed = [doc for doc in documents if not doc.success]

    print(f"Total images processed: {len(documents)}")
    print(f"Successful extractions: {len(successful)}")
    print(f"Failed extractions: {len(failed)}\n")

    for doc in successful:
        print(f"\n{'─'*80}")
        print(f"FILE: {doc.filename}")
        print(f"PATH: {doc.file_path}")
        print(f"TIME: {doc.extraction_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Display quality metrics if available
        if doc.quality_grade:
            quality_display = f"QUALITY: {doc.quality_grade}"
            if doc.quality_score:
                quality_display += f" ({doc.quality_score:.3f})"
            print(quality_display)

        # Display summary if available
        if doc.summary:
            print(f"SUMMARY: {doc.summary}")

        print(f"{'─'*80}")

        # Show first 500 characters of extracted text
        preview_text = doc.extracted_text[:500]
        print(preview_text)

        if len(doc.extracted_text) > 500:
            remaining = len(doc.extracted_text) - 500
            print(f"\n... ({remaining} more characters)")
        print()

    if failed:
        print(f"\n{'─'*80}")
        print("FAILED EXTRACTIONS:")
        print(f"{'─'*80}")
        for doc in failed:
            print(f"  - {doc.filename}: {doc.error_message}")
        print()


def main():
    """Main application entry point."""
    args = parse_arguments()

    try:
        # Step 1: Discover images in the directory
        print(f"Scanning directory: {args.directory}")
        loader = ImageLoader(directory=args.directory)
        image_paths = loader.get_image_paths()
        print(f"Found {len(image_paths)} image(s)\n")

        # Step 2: Extract text from images using Docling with validation
        print("Extracting text from images...")
        valid_image_config = ValidImageConfig(
            min_text_length=args.min_text_length,
            require_heading=args.require_heading,
            min_heading_count=args.min_heading_count,
        )
        extractor = DocumentExtractor(valid_image_config=valid_image_config)
        documents = extractor.extract_batch(image_paths)

        # Step 3: Summarize extracted text using SmolLM3
        print("\nGenerating summaries...")
        summarizer = TextSummarizer()
        documents = summarizer.summarize_batch(documents)

        # Step 4: Display results
        print_results(documents)

        # TODO: Step 5 - Write output to file (not implemented yet)
        # from src.conference_reader.output import OutputWriter
        # writer = OutputWriter(format="md")
        # writer.write_output(documents)

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
