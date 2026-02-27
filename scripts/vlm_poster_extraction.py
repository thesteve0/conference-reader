"""EasyOCR Direct Pipeline for Conference Poster Text Extraction

This script uses EasyOCR directly (without Docling's layout pipeline)
to extract text from conference poster images. This approach is faster
because it skips the layout detection model which can be slow.

Usage:
    python scripts/vlm_poster_extraction.py
    python scripts/vlm_poster_extraction.py -d /path/to/images

References:
    - EasyOCR: https://github.com/JaidedAI/EasyOCR
"""

# =============================================================================
# Environment setup - must be done before importing torch
# =============================================================================
import os

# Fix deprecated TRANSFORMERS_CACHE - use HF_HOME instead (Transformers v5+)
if "TRANSFORMERS_CACHE" in os.environ:
    cache_dir = os.environ.pop("TRANSFORMERS_CACHE")
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = cache_dir
elif "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# Fix deprecated PYTORCH_CUDA_ALLOC_CONF
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    val = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
    os.environ["PYTORCH_ALLOC_CONF"] = val

# Disable cudnn/MIOpen before importing torch
# This avoids workspace allocation issues on ROCm
import torch  # noqa: E402
torch.backends.cudnn.enabled = False

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import easyocr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_IMAGE_DIRECTORY = "/data/neurips/poster_test"
DEFAULT_OUTPUT_FILE = "output/ocr_extracted_posters.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# Image processing settings
# Scale factor for images (1.0 = original, 2.0 = double size)
# Larger images may improve OCR accuracy for small text but take longer
IMAGE_SCALE = 1.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OCRExtractionResult:
    """Result of OCR text extraction from a single image."""

    filename: str
    file_path: str
    extracted_text: str
    extraction_time: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    success: bool = True
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    original_size: Optional[tuple] = None
    processed_size: Optional[tuple] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# GPU Detection
# =============================================================================


def detect_gpu() -> str:
    """Detect available GPU type and return description."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"GPU: {device_name}"
    return "CPU only"


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available()


# =============================================================================
# Image Processing
# =============================================================================


def scale_image(image: Image.Image, scale: float) -> Image.Image:
    """Scale image by a factor.

    Args:
        image: PIL Image to scale
        scale: Scale factor (1.0 = no change, 2.0 = double size)

    Returns:
        Scaled image or original if scale is 1.0
    """
    if scale == 1.0:
        return image

    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.LANCZOS)


# =============================================================================
# Image Discovery
# =============================================================================


def discover_images(directory: str) -> list[str]:
    """Find all image files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {directory}")

    image_paths = []
    for file_path in dir_path.iterdir():
        is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
        if file_path.is_file() and is_image:
            image_paths.append(str(file_path.absolute()))

    return sorted(image_paths)


# =============================================================================
# Text Extraction
# =============================================================================


def extract_single(
    reader: easyocr.Reader,
    image_path: str,
    scale: float = IMAGE_SCALE,
) -> OCRExtractionResult:
    """Extract text from a single image using EasyOCR.

    Args:
        reader: Initialized EasyOCR Reader
        image_path: Path to the image file
        scale: Scale factor for image (1.0 = original, 2.0 = double)

    Returns:
        OCRExtractionResult with extracted text
    """
    start_time = time.time()
    filename = Path(image_path).name

    try:
        # Load and optionally scale image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        image = scale_image(image, scale)
        processed_size = image.size

        # Convert PIL Image to numpy array for EasyOCR
        image_array = np.array(image)

        # Run OCR
        results = reader.readtext(image_array)

        # Extract text from results
        # Each result is (bbox, text, confidence)
        extracted_text = "\n".join([r[1] for r in results])

        processing_time = time.time() - start_time

        return OCRExtractionResult(
            filename=filename,
            file_path=str(Path(image_path).absolute()),
            extracted_text=extracted_text,
            success=True,
            processing_time_seconds=round(processing_time, 2),
            original_size=original_size,
            processed_size=processed_size,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return OCRExtractionResult(
            filename=filename,
            file_path=str(Path(image_path).absolute()),
            extracted_text="",
            success=False,
            error_message=str(e),
            processing_time_seconds=round(processing_time, 2),
        )


def extract_batch(
    image_paths: list[str],
    use_gpu: bool = True,
    scale: float = IMAGE_SCALE,
    verbose: bool = True,
) -> list[OCRExtractionResult]:
    """Extract text from multiple images using EasyOCR.

    Args:
        image_paths: List of paths to image files
        use_gpu: Whether to use GPU acceleration
        scale: Scale factor for images (1.0 = original, 2.0 = double)
        verbose: Print progress information

    Returns:
        List of OCRExtractionResult objects
    """
    results = []
    total = len(image_paths)

    # Initialize EasyOCR reader
    if verbose:
        gpu_status = "GPU" if use_gpu and is_gpu_available() else "CPU"
        print(f"Initializing EasyOCR ({gpu_status})...")

    reader = easyocr.Reader(["en"], gpu=use_gpu and is_gpu_available())

    for i, path in enumerate(image_paths, start=1):
        filename = Path(path).name

        if verbose:
            print(f"\n[{i}/{total}] Processing: {filename}")

        result = extract_single(reader, path, scale)
        results.append(result)

        if verbose:
            if result.success:
                size_info = ""
                if result.original_size != result.processed_size:
                    orig = result.original_size
                    proc = result.processed_size
                    size_info = f" (resized {orig} -> {proc})"
                t = result.processing_time_seconds
                print(f"  Success ({t}s){size_info}")
                print(f"  Extracted {len(result.extracted_text)} chars")
                if result.extracted_text:
                    preview = result.extracted_text[:100].replace("\n", " ")
                    if len(result.extracted_text) > 100:
                        preview += "..."
                    print(f"  Preview: {preview}")
            else:
                print(f"  Failed: {result.error_message}")

    return results


# =============================================================================
# Output
# =============================================================================


def save_results(results: list[OCRExtractionResult], output_path: str) -> None:
    """Save extraction results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "extraction_date": datetime.now().isoformat(),
        "total_images": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "ocr_engine": "easyocr",
        "image_scale": IMAGE_SCALE,
        "results": [r.to_dict() for r in results],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(results: list[OCRExtractionResult]) -> None:
    """Print extraction summary statistics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        times = [
            r.processing_time_seconds
            for r in successful
            if r.processing_time_seconds
        ]
        if times:
            avg_time = sum(times) / len(times)
            print(f"Average processing time: {avg_time:.2f}s")
            print(f"Total processing time: {sum(times):.2f}s")

        total_chars = sum(len(r.extracted_text) for r in successful)
        print(f"Total characters extracted: {total_chars}")

    if failed:
        print("\nFailed extractions:")
        for r in failed:
            print(f"  - {r.filename}: {r.error_message}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract text from conference poster images using EasyOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/vlm_poster_extraction.py
    python scripts/vlm_poster_extraction.py -d /data/posters
    python scripts/vlm_poster_extraction.py --scale 2.0  # Enlarge images 2x for small text
        """,
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=DEFAULT_IMAGE_DIRECTORY,
        help=f"Directory with poster images (default: {DEFAULT_IMAGE_DIRECTORY})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=IMAGE_SCALE,
        help=f"Image scale factor (default: {IMAGE_SCALE}). Use >1.0 to enlarge images for better OCR of small text.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    verbose = not args.quiet
    use_gpu = not args.no_gpu
    scale = args.scale

    if verbose:
        print("=" * 60)
        print("EASYOCR POSTER TEXT EXTRACTION")
        print("=" * 60)
        print(f"Hardware: {detect_gpu()}")
        print(f"GPU Enabled: {use_gpu and is_gpu_available()}")
        print("cuDNN/MIOpen: Disabled (for ROCm compatibility)")
        if scale != 1.0:
            print(f"Image scale factor: {scale}x")
        else:
            print("Image scale: Original size (1.0x)")
        print(f"Input directory: {args.directory}")
        print(f"Output file: {args.output}")

    try:
        # Discover images
        if verbose:
            print(f"\nScanning for images in: {args.directory}")
        image_paths = discover_images(args.directory)

        if not image_paths:
            print("No images found. Exiting.")
            sys.exit(0)

        if verbose:
            print(f"Found {len(image_paths)} image(s)")

        # Extract text
        results = extract_batch(
            image_paths,
            use_gpu=use_gpu,
            scale=scale,
            verbose=verbose,
        )

        # Save results
        save_results(results, args.output)
        if verbose:
            print(f"\nResults saved to: {args.output}")

        # Print summary
        print_summary(results)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
