"""DocumentExtractor class for extracting text from images using EasyOCR."""

# =============================================================================
# Environment setup - must be done before importing torch (via easyocr)
# =============================================================================
import os

# Fix deprecated PYTORCH_CUDA_ALLOC_CONF before torch imports
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    val = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = val

# =============================================================================
# Standard imports
# =============================================================================
import gc
import time
from pathlib import Path
from typing import List, Optional

import easyocr
import numpy as np
from PIL import Image

from ..config.rocm_config import apply_rocm_stability_settings
from .processed_document import ProcessedDocument

# Threshold for considering an extraction "slow" (seconds)
SLOW_EXTRACTION_THRESHOLD = 60.0


def _is_gpu_available() -> bool:
    """Check if GPU is available for EasyOCR."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _disable_cudnn() -> None:
    """Disable cuDNN/MIOpen for ROCm compatibility."""
    try:
        import torch

        torch.backends.cudnn.enabled = False
    except ImportError:
        pass


class DocumentExtractor:
    """Extracts text from images using EasyOCR.

    This class wraps EasyOCR to provide a clean interface
    for extracting text from conference poster images.

    Note:
        Image filtering (poster vs QR code) should be done BEFORE calling
        this extractor using the ImageClassifier class.

    Attributes:
        reader: EasyOCR Reader instance
        images_scale: Scale factor for image resolution
        document_timeout: Maximum seconds per document (not enforced by EasyOCR)
    """

    def __init__(
        self,
        images_scale: float = 1.0,
        document_timeout: Optional[float] = 120.0,
        use_gpu: bool = True,
    ):
        """Initialize DocumentExtractor with configuration options.

        Args:
            images_scale: Scale factor for image resolution (default: 1.0).
                0.5 = half resolution, 1.0 = full resolution, 2.0 = double.
            document_timeout: Maximum seconds per document (default: 120).
                Note: EasyOCR doesn't support timeouts, kept for API compatibility.
            use_gpu: Whether to use GPU acceleration (default: True).
        """
        # Apply ROCm stability settings for AMD GPUs before any GPU operations
        if use_gpu:
            apply_rocm_stability_settings()

        # Disable cuDNN/MIOpen for ROCm compatibility
        _disable_cudnn()

        self.images_scale = images_scale
        self.document_timeout = document_timeout
        self.use_gpu = use_gpu and _is_gpu_available()
        self.reader = self._create_reader()

    def _create_reader(self) -> easyocr.Reader:
        """Create a fresh EasyOCR Reader with current settings.

        Returns:
            Configured EasyOCR Reader instance
        """
        return easyocr.Reader(["en"], gpu=self.use_gpu)

    def _scale_image(self, image: Image.Image) -> Image.Image:
        """Scale image by the configured factor.

        Args:
            image: PIL Image to scale

        Returns:
            Scaled image or original if scale is 1.0
        """
        if self.images_scale == 1.0:
            return image

        width, height = image.size
        new_width = int(width * self.images_scale)
        new_height = int(height * self.images_scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def _reset_reader(self, verbose: bool = False) -> None:
        """Reset the reader to recover from issues.

        This cleans up GPU memory and creates a fresh reader instance.
        Call this after slow extractions or failures to prevent resource leaks.

        Args:
            verbose: If True, print a message about the reset
        """
        if verbose:
            print("  -> Resetting reader to recover resources...")

        # Delete old reader
        del self.reader

        # Force garbage collection to release GPU memory
        gc.collect()

        # Try to clear GPU cache if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass  # torch not available or no GPU

        # Create fresh reader
        self.reader = self._create_reader()

    def extract_single(self, image_path: str) -> ProcessedDocument:
        """Extract text from a single image file.

        Args:
            image_path: Path to the image file to process

        Returns:
            ProcessedDocument instance with extracted text and timing
        """
        start_time = time.time()

        try:
            # Load and optionally scale image
            image = Image.open(image_path).convert("RGB")
            image = self._scale_image(image)

            # Convert PIL Image to numpy array for EasyOCR
            image_array = np.array(image)

            # Run OCR
            results = self.reader.readtext(image_array)

            processing_time = time.time() - start_time

            # Extract text from results
            # Each result is (bbox, text, confidence)
            extracted_text = "\n".join([r[1] for r in results])

            return ProcessedDocument.from_path(
                file_path=image_path,
                extracted_text=extracted_text,
                success=True,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            return ProcessedDocument.from_error(
                file_path=image_path,
                error_message=error_msg,
                processing_time=processing_time,
            )

    def extract_batch(
        self,
        image_paths: List[str],
        verbose: bool = False,
    ) -> List[ProcessedDocument]:
        """Extract text from multiple image files.

        After slow extractions (>60s) or failures, the reader is reset
        to recover GPU resources and prevent issues from accumulating.

        Args:
            image_paths: List of paths to image files to process
            verbose: If True, print progress for each image

        Returns:
            List of ProcessedDocument instances, one per input image
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths, start=1):
            filename = Path(path).name
            print(f"DocumentExtractor: {filename}")

            if verbose:
                print(f"Processing [{i}/{total}]: {filename}")

            doc = self.extract_single(path)
            results.append(doc)
            if doc.processing_time is not None:
                print(f"{filename} took {doc.processing_time:.2f} seconds\n")

            if verbose:
                time_str = (
                    f"{doc.processing_time:.2f}s\n" if doc.processing_time else "N/A"
                )
                if doc.success:
                    print(f"  -> Success ({time_str})")
                else:
                    print(f"  -> Failed: {doc.error_message} ({time_str})")

            # Reset reader after slow extractions or failures to prevent
            # issues from accumulating and exhausting GPU resources
            needs_reset = not doc.success or (
                doc.processing_time is not None
                and doc.processing_time > SLOW_EXTRACTION_THRESHOLD
            )
            if needs_reset:
                self._reset_reader(verbose=verbose)

        return results
