"""Tests for DocumentExtractor class."""

from unittest.mock import Mock, patch, MagicMock
import numpy as np

import pytest

from conference_reader.extraction import DocumentExtractor, ProcessedDocument


class TestDocumentExtractorWithMock:
    """Tests for DocumentExtractor using mocked EasyOCR."""

    @pytest.fixture
    def mock_extractor(self):
        """Create extractor with mocked EasyOCR Reader."""
        with patch(
            "conference_reader.extraction.document_extractor.easyocr.Reader"
        ) as MockReader:
            mock_reader = Mock()
            MockReader.return_value = mock_reader
            extractor = DocumentExtractor()
            extractor.reader = mock_reader
            yield extractor, mock_reader

    @pytest.fixture
    def mock_extractor_custom(self):
        """Create extractor with custom scale and timeout."""
        with patch(
            "conference_reader.extraction.document_extractor.easyocr.Reader"
        ) as MockReader:
            mock_reader = Mock()
            MockReader.return_value = mock_reader
            extractor = DocumentExtractor(images_scale=0.5, document_timeout=60.0)
            extractor.reader = mock_reader
            yield extractor, mock_reader

    def test_constructor_default_values(self):
        """Test constructor uses default values."""
        with patch(
            "conference_reader.extraction.document_extractor.easyocr.Reader"
        ):
            extractor = DocumentExtractor()
            assert extractor.images_scale == 1.0
            assert extractor.document_timeout == 120.0

    def test_constructor_custom_values(self):
        """Test constructor accepts custom values."""
        with patch(
            "conference_reader.extraction.document_extractor.easyocr.Reader"
        ):
            extractor = DocumentExtractor(images_scale=0.5, document_timeout=60.0)
            assert extractor.images_scale == 0.5
            assert extractor.document_timeout == 60.0

    def test_extract_single_success(self, mock_extractor):
        """Test successful extraction of a single document."""
        extractor, mock_reader = mock_extractor

        # Mock the EasyOCR result (list of (bbox, text, confidence) tuples)
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Test Title", 0.95),
            ([[0, 30], [100, 30], [100, 50], [0, 50]], "Content", 0.90),
        ]

        # Mock PIL Image.open
        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image.size = (100, 100)
            mock_open.return_value = mock_image

            doc = extractor.extract_single("/test/image.jpg")

        assert isinstance(doc, ProcessedDocument)
        assert doc.success is True
        assert "Test Title" in doc.extracted_text
        assert "Content" in doc.extracted_text

    def test_extract_single_failure(self, mock_extractor):
        """Test handling of extraction failure."""
        extractor, mock_reader = mock_extractor

        # Mock PIL Image.open to raise an exception
        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_open.side_effect = Exception("Processing failed")

            doc = extractor.extract_single("/test/bad_image.jpg")

        assert isinstance(doc, ProcessedDocument)
        assert doc.success is False
        assert "Processing failed" in doc.error_message

    def test_extract_batch(self, mock_extractor):
        """Test batch extraction of multiple documents."""
        extractor, mock_reader = mock_extractor

        # Mock successful results
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.95),
            ([[0, 30], [100, 30], [100, 50], [0, 50]], "Text", 0.90),
        ]

        # Mock PIL Image.open
        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image.size = (100, 100)
            mock_open.return_value = mock_image

            paths = ["/test/img1.jpg", "/test/img2.jpg", "/test/img3.jpg"]
            docs = extractor.extract_batch(paths)

        assert len(docs) == 3
        assert all(doc.success for doc in docs)
        assert mock_reader.readtext.call_count == 3

    def test_extract_single_includes_processing_time(self, mock_extractor):
        """Test that extraction includes processing_time."""
        extractor, mock_reader = mock_extractor

        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.95),
        ]

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image.size = (100, 100)
            mock_open.return_value = mock_image

            doc = extractor.extract_single("/test/image.jpg")

        assert doc.processing_time is not None
        assert doc.processing_time >= 0

    def test_extract_single_failure_includes_processing_time(self, mock_extractor):
        """Test that failed extraction includes processing_time."""
        extractor, mock_reader = mock_extractor

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_open.side_effect = Exception("Processing failed")

            doc = extractor.extract_single("/test/bad_image.jpg")

        assert doc.success is False
        assert doc.processing_time is not None
        assert doc.processing_time >= 0

    def test_extract_batch_verbose_output(self, mock_extractor, capsys):
        """Test verbose output during batch extraction."""
        extractor, mock_reader = mock_extractor

        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.95),
        ]

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image.size = (100, 100)
            mock_open.return_value = mock_image

            paths = ["/test/img1.jpg", "/test/img2.jpg"]
            extractor.extract_batch(paths, verbose=True)

        captured = capsys.readouterr()
        assert "Processing [1/2]: img1.jpg" in captured.out
        assert "Processing [2/2]: img2.jpg" in captured.out
        assert "Success" in captured.out

    def test_extract_batch_verbose_shows_failures(self, mock_extractor, capsys):
        """Test verbose output shows failure messages."""
        extractor, mock_reader = mock_extractor

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_open.side_effect = Exception("Timeout after 120s")

            paths = ["/test/img1.jpg"]
            extractor.extract_batch(paths, verbose=True)

        captured = capsys.readouterr()
        assert "Processing [1/1]: img1.jpg" in captured.out
        assert "Failed" in captured.out

    def test_extract_batch_resets_after_slow_extraction(
        self, mock_extractor, capsys
    ):
        """Test that reader is reset after slow extractions."""
        extractor, mock_reader = mock_extractor

        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.95),
        ]

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image.size = (100, 100)
            mock_open.return_value = mock_image

            # Patch time.time to simulate slow extraction (>60s)
            time_module = "conference_reader.extraction.document_extractor.time"
            with patch(time_module) as mock_time:
                # First call returns start time, second returns 70 seconds later
                mock_time.time.side_effect = [0, 70, 0, 5]

                with patch.object(extractor, "_reset_reader") as mock_reset:
                    paths = ["/test/slow_img.jpg", "/test/fast_img.jpg"]
                    extractor.extract_batch(paths, verbose=False)

                    # Should reset after slow extraction
                    assert mock_reset.call_count == 1

    def test_extract_batch_resets_after_failure(self, mock_extractor, capsys):
        """Test that reader is reset after failed extractions."""
        extractor, mock_reader = mock_extractor

        with patch(
            "conference_reader.extraction.document_extractor.Image.open"
        ) as mock_open:
            mock_open.side_effect = Exception("Processing failed")

            with patch.object(extractor, "_reset_reader") as mock_reset:
                paths = ["/test/bad_img.jpg"]
                extractor.extract_batch(paths, verbose=False)

                # Should reset after failure
                assert mock_reset.call_count == 1
