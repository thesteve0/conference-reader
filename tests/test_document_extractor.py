"""Tests for DocumentExtractor class."""

from unittest.mock import Mock, patch

import pytest

from conference_reader.extraction import DocumentExtractor, ProcessedDocument


class TestDocumentExtractorTitleExtraction:
    """Tests for title extraction logic (no external deps)."""

    def test_extract_title_from_heading(self):
        """Test extracting title from markdown heading."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("# My Poster Title\n\nContent here")
        assert title == "My Poster Title"

    def test_extract_title_from_h2(self):
        """Test extracting title from h2 heading."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("## Secondary Title\n\nContent")
        assert title == "Secondary Title"

    def test_extract_title_no_heading(self):
        """Test extracting title when no heading symbol."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("Plain text title\n\nContent")
        assert title == "Plain text title"

    def test_extract_title_empty_text(self):
        """Test extracting title from empty text."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("")
        assert title is None

    def test_extract_title_only_whitespace(self):
        """Test extracting title from whitespace-only text."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("   \n\n   ")
        assert title is None

    def test_extract_title_skips_empty_lines(self):
        """Test that title extraction skips leading empty lines."""
        extractor = DocumentExtractor.__new__(DocumentExtractor)

        title = extractor._extract_title("\n\n# Actual Title\n\nContent")
        assert title == "Actual Title"


class TestDocumentExtractorWithMock:
    """Tests for DocumentExtractor using mocked Docling."""

    @pytest.fixture
    def mock_extractor(self):
        """Create extractor with mocked DocumentConverter."""
        with patch(
            "conference_reader.extraction.document_extractor.DocumentConverter"
        ) as MockConverter:
            mock_converter = Mock()
            MockConverter.return_value = mock_converter
            extractor = DocumentExtractor()
            extractor.converter = mock_converter
            yield extractor, mock_converter

    @pytest.fixture
    def mock_extractor_custom(self):
        """Create extractor with custom scale and timeout."""
        with patch(
            "conference_reader.extraction.document_extractor.DocumentConverter"
        ) as MockConverter:
            mock_converter = Mock()
            MockConverter.return_value = mock_converter
            extractor = DocumentExtractor(images_scale=0.5, document_timeout=60.0)
            extractor.converter = mock_converter
            yield extractor, mock_converter

    def test_constructor_default_values(self):
        """Test constructor uses default values."""
        with patch(
            "conference_reader.extraction.document_extractor.DocumentConverter"
        ):
            extractor = DocumentExtractor()
            assert extractor.images_scale == 0.75
            assert extractor.document_timeout == 120.0

    def test_constructor_custom_values(self):
        """Test constructor accepts custom values."""
        with patch(
            "conference_reader.extraction.document_extractor.DocumentConverter"
        ):
            extractor = DocumentExtractor(images_scale=0.5, document_timeout=60.0)
            assert extractor.images_scale == 0.5
            assert extractor.document_timeout == 60.0

    def test_extract_single_success(self, mock_extractor):
        """Test successful extraction of a single document."""
        extractor, mock_converter = mock_extractor

        # Mock the Docling result
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = (
            "# Test Title\n\nContent"
        )
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        doc = extractor.extract_single("/test/image.jpg")

        assert isinstance(doc, ProcessedDocument)
        assert doc.success is True
        assert doc.title == "Test Title"
        assert doc.extracted_text == "# Test Title\n\nContent"
        assert doc.quality_grade == "GOOD"
        assert doc.quality_score == 0.85

    def test_extract_single_failure(self, mock_extractor):
        """Test handling of extraction failure."""
        extractor, mock_converter = mock_extractor
        mock_converter.convert.side_effect = Exception("Processing failed")

        doc = extractor.extract_single("/test/bad_image.jpg")

        assert isinstance(doc, ProcessedDocument)
        assert doc.success is False
        assert "Processing failed" in doc.error_message

    def test_extract_batch(self, mock_extractor):
        """Test batch extraction of multiple documents."""
        extractor, mock_converter = mock_extractor

        # Mock successful results
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Title\n\nText"
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        paths = ["/test/img1.jpg", "/test/img2.jpg", "/test/img3.jpg"]
        docs = extractor.extract_batch(paths)

        assert len(docs) == 3
        assert all(doc.success for doc in docs)
        assert mock_converter.convert.call_count == 3

    def test_extract_single_includes_processing_time(self, mock_extractor):
        """Test that extraction includes processing_time."""
        extractor, mock_converter = mock_extractor

        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Title\n\nText"
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        doc = extractor.extract_single("/test/image.jpg")

        assert doc.processing_time is not None
        assert doc.processing_time >= 0

    def test_extract_single_failure_includes_processing_time(self, mock_extractor):
        """Test that failed extraction includes processing_time."""
        extractor, mock_converter = mock_extractor
        mock_converter.convert.side_effect = Exception("Processing failed")

        doc = extractor.extract_single("/test/bad_image.jpg")

        assert doc.success is False
        assert doc.processing_time is not None
        assert doc.processing_time >= 0

    def test_extract_batch_verbose_output(self, mock_extractor, capsys):
        """Test verbose output during batch extraction."""
        extractor, mock_converter = mock_extractor

        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Title\n\nText"
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        paths = ["/test/img1.jpg", "/test/img2.jpg"]
        extractor.extract_batch(paths, verbose=True)

        captured = capsys.readouterr()
        assert "Processing [1/2]: img1.jpg" in captured.out
        assert "Processing [2/2]: img2.jpg" in captured.out
        assert "Success" in captured.out

    def test_extract_batch_verbose_shows_failures(self, mock_extractor, capsys):
        """Test verbose output shows failure messages."""
        extractor, mock_converter = mock_extractor
        mock_converter.convert.side_effect = Exception("Timeout after 120s")

        paths = ["/test/img1.jpg"]
        extractor.extract_batch(paths, verbose=True)

        captured = capsys.readouterr()
        assert "Processing [1/1]: img1.jpg" in captured.out
        assert "Failed" in captured.out

    def test_extract_batch_silent_by_default(self, mock_extractor, capsys):
        """Test that batch extraction is silent when verbose=False."""
        extractor, mock_converter = mock_extractor

        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Title\n\nText"
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        paths = ["/test/img1.jpg"]
        extractor.extract_batch(paths, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_extract_batch_resets_after_slow_extraction(
        self, mock_extractor, capsys
    ):
        """Test that converter is reset after slow extractions."""
        extractor, mock_converter = mock_extractor

        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Title\n\nText"
        mock_result.confidence.mean_grade = "GOOD"
        mock_result.confidence.mean_score = 0.85
        mock_result.confidence.low_grade = "FAIR"
        mock_result.confidence.low_score = 0.7
        mock_result.confidence.ocr_score = 0.9
        mock_result.confidence.layout_score = 0.8
        mock_converter.convert.return_value = mock_result

        # Patch time.time to simulate slow extraction (>60s)
        time_module = "conference_reader.extraction.document_extractor.time"
        with patch(time_module) as mock_time:
            # First call returns start time, second returns 70 seconds later
            mock_time.time.side_effect = [0, 70, 0, 5]

            with patch.object(extractor, "_reset_converter") as mock_reset:
                paths = ["/test/slow_img.jpg", "/test/fast_img.jpg"]
                extractor.extract_batch(paths, verbose=False)

                # Should reset after slow extraction
                assert mock_reset.call_count == 1

    def test_extract_batch_resets_after_failure(self, mock_extractor, capsys):
        """Test that converter is reset after failed extractions."""
        extractor, mock_converter = mock_extractor
        mock_converter.convert.side_effect = Exception("Processing failed")

        with patch.object(extractor, "_reset_converter") as mock_reset:
            paths = ["/test/bad_img.jpg"]
            extractor.extract_batch(paths, verbose=False)

            # Should reset after failure
            assert mock_reset.call_count == 1
