"""DocumentExtractor class for extracting text from images using Docling."""

from typing import List

from docling.document_converter import DocumentConverter

from .processed_document import ProcessedDocument
from .valid_image_config import ValidImageConfig


class DocumentExtractor:
    """Extracts text from images using Docling's DocumentConverter.

    This class wraps Docling's DocumentConverter to provide a clean interface
    for extracting text from conference poster images.

    Attributes:
        converter: Docling's DocumentConverter instance
    """

    def __init__(self, valid_image_config: ValidImageConfig):
        """Initialize DocumentExtractor with validation config.

        Args:
            valid_image_config: Configuration for validation thresholds.
        """
        self.converter = DocumentConverter()
        self.valid_image_config = valid_image_config

    def extract_single(self, image_path: str) -> ProcessedDocument:
        """Extract text from a single image file.

        Args:
            image_path: Path to the image file to process

        Returns:
            ProcessedDocument instance with extracted text
        """
        try:
            # Convert the image using Docling
            result = self.converter.convert(image_path)

            # Extract the text as markdown
            extracted_text = result.document.export_to_markdown()

            # Extract quality metrics from Docling's confidence scores
            confidence = result.confidence

            # Extract document structure metadata
            children_count = (
                len(result.document.body.children)
                if hasattr(result.document, "body")
                and hasattr(result.document.body, "children")
                else 0
            )

            # Validate poster quality
            is_valid, error_msg = self._validate_poster(
                extracted_text, children_count
            )

            if not is_valid:
                # Return as extraction failure with validation error message
                return ProcessedDocument.from_error(
                    file_path=image_path, error_message=error_msg
                )

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
            )

        except Exception as e:
            # Return error document if extraction fails
            return ProcessedDocument.from_error(
                file_path=image_path, error_message=str(e)
            )

    def _validate_poster(
        self, extracted_text: str, children_count: int
    ) -> tuple[bool, str]:
        """Validate extracted content represents a complete poster.

        Checks text-based and structural heuristics to determine if
        extracted content is valid or should be filtered (QR codes,
        partial crops, code snippets).

        Args:
            extracted_text: Extracted markdown text from Docling
            children_count: Number of document structure children elements

        Returns:
            (is_valid, error_message) tuple. Empty string if valid.

        Validation Logic:
            Image fails ONLY if ALL checks fail (AND logic).
            Permissive approach reduces false negatives.
        """
        config = self.valid_image_config
        failures = []
        total_checks = 0

        # Check text length
        total_checks += 1
        text_len = len(extracted_text.strip())
        if text_len < config.min_text_length:
            failures.append(
                f"text too short ({text_len} chars < "
                f"{config.min_text_length})"
            )

        # Count markdown headings in the entire document
        heading_count = sum(
            1
            for line in extracted_text.split("\n")
            if line.strip().startswith("#")
        )

        # Check for markdown heading at the beginning (title check)
        if config.require_heading:
            total_checks += 1
            preview = extracted_text[:500]
            has_heading = any(
                line.strip().startswith("#")
                for line in preview.split("\n")
            )
            if not has_heading:
                failures.append(
                    "no markdown heading found (expected title)"
                )

        # Check total heading count
        total_checks += 1
        if heading_count < config.min_heading_count:
            failures.append(
                f"too few headings ({heading_count} < "
                f"{config.min_heading_count})"
            )

        # Check document structure complexity
        total_checks += 1
        if children_count < config.min_children_count:
            failures.append(
                f"too few children ({children_count} < "
                f"{config.min_children_count})"
            )

        # Only mark as invalid if ALL checks failed
        if len(failures) == total_checks:
            return False, f"Invalid poster: {'; '.join(failures)}"

        return True, ""

    def extract_batch(self, image_paths: List[str]) -> List[ProcessedDocument]:
        """Extract text from multiple image files.

        Args:
            image_paths: List of paths to image files to process

        Returns:
            List of ProcessedDocument instances, one per input image
        """
        return [self.extract_single(path) for path in image_paths]
