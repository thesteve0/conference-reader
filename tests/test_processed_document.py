"""Tests for ProcessedDocument dataclass."""

import pytest

from conference_reader.extraction import ProcessedDocument


class TestProcessedDocument:
    """Tests for ProcessedDocument dataclass."""

    def test_from_path_creates_document(self):
        """Test that from_path creates a valid document."""
        doc = ProcessedDocument.from_path(
            file_path="/data/test/image.jpg",
            extracted_text="# Test Title\n\nContent",
        )

        assert doc.filename == "image.jpg"
        assert doc.file_path.endswith("image.jpg")
        assert doc.extracted_text == "# Test Title\n\nContent"
        assert doc.success is True
        assert doc.error_message is None

    def test_from_path_with_quality_metrics(self):
        """Test that from_path handles quality metrics."""
        doc = ProcessedDocument.from_path(
            file_path="/data/test/image.jpg",
            extracted_text="content",
            quality_grade="GOOD",
            quality_score=0.85,
            ocr_score=0.9,
            layout_score=0.8,
        )

        assert doc.quality_grade == "GOOD"
        assert doc.quality_score == 0.85
        assert doc.ocr_score == 0.9
        assert doc.layout_score == 0.8

    def test_from_error_creates_failed_document(self):
        """Test that from_error creates a failed document."""
        doc = ProcessedDocument.from_error(
            file_path="/data/test/bad.jpg",
            error_message="Failed to process image",
        )

        assert doc.filename == "bad.jpg"
        assert doc.success is False
        assert doc.error_message == "Failed to process image"
        assert doc.extracted_text == ""

    def test_default_values(self):
        """Test that optional fields have correct defaults."""
        doc = ProcessedDocument(
            filename="test.jpg",
            file_path="/test.jpg",
            extracted_text="content",
        )

        assert doc.success is True
        assert doc.error_message is None
        assert doc.quality_grade is None
        assert doc.quality_score is None
        assert doc.summary is None

    def test_extraction_time_is_set(self):
        """Test that extraction_time is automatically set."""
        doc = ProcessedDocument(
            filename="test.jpg",
            file_path="/test.jpg",
            extracted_text="content",
        )

        assert doc.extraction_time is not None
