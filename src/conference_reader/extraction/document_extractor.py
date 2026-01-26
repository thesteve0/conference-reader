"""DocumentExtractor class for extracting text from images using Docling."""

import gc
import time
from pathlib import Path
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .processed_document import ProcessedDocument

# Threshold for considering an extraction "slow" (seconds)
SLOW_EXTRACTION_THRESHOLD = 60.0


class DocumentExtractor:
    """Extracts text from images using Docling's DocumentConverter.

    This class wraps Docling's DocumentConverter to provide a clean interface
    for extracting text from conference poster images.

    Note:
        Image filtering (poster vs QR code) should be done BEFORE calling
        this extractor using the ImageClassifier class.

    Attributes:
        converter: Docling's DocumentConverter instance
        images_scale: Scale factor for image resolution
        document_timeout: Maximum seconds per document
    """

    def __init__(
        self,
        images_scale: float = 0.75,
        document_timeout: Optional[float] = 120.0,
    ):
        """Initialize DocumentExtractor with configuration options.

        Args:
            images_scale: Scale factor for image resolution (default: 0.75).
                0.5 = half resolution, 1.0 = full resolution, 2.0 = double.
            document_timeout: Maximum seconds per document (default: 120).
                None = no timeout.
        """
        self.images_scale = images_scale
        self.document_timeout = document_timeout
        self.converter = self._create_converter()

    def _create_converter(self) -> DocumentConverter:
        """Create a fresh DocumentConverter with current settings.

        Returns:
            Configured DocumentConverter instance
        """
        pipeline_options = PdfPipelineOptions(
            images_scale=self.images_scale,
            document_timeout=self.document_timeout,
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )

    def _reset_converter(self, verbose: bool = False) -> None:
        """Reset the converter to recover from stuck threads.

        This cleans up GPU memory and creates a fresh converter instance.
        Call this after timeouts or slow extractions to prevent resource leaks.

        Args:
            verbose: If True, print a message about the reset
        """
        if verbose:
            print("  -> Resetting converter to recover resources...")

        # Delete old converter
        del self.converter

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

        # Create fresh converter
        self.converter = self._create_converter()

    def _extract_title(self, extracted_text: str) -> Optional[str]:
        """Extract title from the first non-empty line of markdown.

        Args:
            extracted_text: Markdown-formatted text from Docling

        Returns:
            Title with markdown symbols stripped, or None if not found
        """
        if not extracted_text:
            return None

        for line in extracted_text.split("\n"):
            stripped = line.strip()
            if stripped:
                # Remove markdown heading symbols and return
                return stripped.lstrip("#").strip()

        return None

    def extract_single(self, image_path: str) -> ProcessedDocument:
        """Extract text from a single image file.

        Args:
            image_path: Path to the image file to process

        Returns:
            ProcessedDocument instance with extracted text, title, and timing
        """
        start_time = time.time()

        try:
            # Convert the image using Docling
            result = self.converter.convert(image_path)
            processing_time = time.time() - start_time

            # Extract the text as markdown
            extracted_text = result.document.export_to_markdown()

            # Extract title from the markdown
            title = self._extract_title(extracted_text)

            # Extract quality metrics from Docling's confidence scores
            confidence = result.confidence

            return ProcessedDocument.from_path(
                file_path=image_path,
                extracted_text=extracted_text,
                success=True,
                quality_grade=str(confidence.mean_grade),
                quality_score=confidence.mean_score,
                low_quality_grade=str(confidence.low_grade),
                low_score=confidence.low_score,
                ocr_score=confidence.ocr_score,
                layout_score=confidence.layout_score,
                title=title,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            # Make timeout errors more readable
            if "timeout" in error_msg.lower():
                error_msg = f"Timeout after {processing_time:.1f}s"

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

        After slow extractions (>60s) or failures, the converter is reset
        to recover GPU resources and prevent stuck threads from accumulating.

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

            if verbose:
                print(f"Processing [{i}/{total}]: {filename}")

            doc = self.extract_single(path)
            results.append(doc)

            if verbose:
                time_str = (
                    f"{doc.processing_time:.2f}s"
                    if doc.processing_time
                    else "N/A"
                )
                if doc.success:
                    print(f"  -> Success ({time_str})")
                else:
                    print(f"  -> Failed: {doc.error_message} ({time_str})")

            # Reset converter after slow extractions or failures to prevent
            # stuck threads from accumulating and exhausting GPU resources
            needs_reset = (
                not doc.success
                or (
                    doc.processing_time is not None
                    and doc.processing_time > SLOW_EXTRACTION_THRESHOLD
                )
            )
            if needs_reset:
                self._reset_converter(verbose=verbose)

        return results
