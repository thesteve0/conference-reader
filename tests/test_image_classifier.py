"""Tests for ImageClassifier class."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from conference_reader.classifier import (
    ImageClassifier,
    ImageType,
    ClassificationResult,
)


class TestImageClassifierParsing:
    """Tests for ImageClassifier response parsing (no GPU needed)."""

    def test_parse_response_poster(self):
        """Test parsing a clear 'poster' response."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)
            image_type, confidence = classifier._parse_response("poster")

            assert image_type == ImageType.POSTER
            assert confidence == 1.0

    def test_parse_response_poster_with_text(self):
        """Test parsing response containing 'poster' with other text."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)
            image_type, confidence = classifier._parse_response(
                "This is a poster image"
            )

            assert image_type == ImageType.POSTER
            assert confidence == 1.0

    def test_parse_response_qr(self):
        """Test parsing a 'qr' response."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)
            image_type, confidence = classifier._parse_response("qr")

            assert image_type == ImageType.QR_CODE
            assert confidence == 1.0

    def test_parse_response_qr_takes_precedence(self):
        """Test that 'qr' takes precedence if both words present."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)
            image_type, confidence = classifier._parse_response(
                "This looks like a poster but has qr codes"
            )

            assert image_type == ImageType.QR_CODE
            assert confidence == 1.0

    def test_parse_response_unknown(self):
        """Test parsing ambiguous response."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)
            image_type, confidence = classifier._parse_response(
                "I cannot determine what this is"
            )

            assert image_type == ImageType.UNKNOWN
            assert confidence == 0.5

    def test_parse_response_case_insensitive(self):
        """Test that parsing is case insensitive."""
        with patch.object(ImageClassifier, "__init__", lambda x: None):
            classifier = ImageClassifier.__new__(ImageClassifier)

            image_type, _ = classifier._parse_response("POSTER")
            assert image_type == ImageType.POSTER

            image_type, _ = classifier._parse_response("QR")
            assert image_type == ImageType.QR_CODE


class TestImageClassifierWithMock:
    """Tests for ImageClassifier using mocked VLMBackend."""

    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked backend."""
        with patch(
            "conference_reader.classifier.image_classifier.VLMBackend"
        ) as MockBackend:
            mock_backend = Mock()
            MockBackend.return_value = mock_backend
            classifier = ImageClassifier()
            classifier._backend = mock_backend
            yield classifier, mock_backend

    def test_classify_returns_result(self, mock_classifier):
        """Test that classify returns a ClassificationResult."""
        classifier, mock_backend = mock_classifier
        mock_backend.generate.return_value = ("poster", 1.5)

        result = classifier.classify(Path("/test/image.jpg"))

        assert isinstance(result, ClassificationResult)
        assert result.image_type == ImageType.POSTER
        assert result.image_path == Path("/test/image.jpg")
        assert result.inference_time == 1.5

    def test_classify_batch(self, mock_classifier):
        """Test classifying multiple images."""
        classifier, mock_backend = mock_classifier
        mock_backend.generate.side_effect = [
            ("poster", 1.0),
            ("qr", 0.8),
            ("poster", 1.2),
        ]

        paths = [
            Path("/test/img1.jpg"),
            Path("/test/img2.jpg"),
            Path("/test/img3.jpg"),
        ]
        results = classifier.classify_batch(paths)

        assert len(results) == 3
        assert results[0].image_type == ImageType.POSTER
        assert results[1].image_type == ImageType.QR_CODE
        assert results[2].image_type == ImageType.POSTER

    def test_filter_posters(self, mock_classifier):
        """Test filtering to only poster images."""
        classifier, mock_backend = mock_classifier
        mock_backend.generate.side_effect = [
            ("poster", 1.0),
            ("qr", 0.8),
            ("poster", 1.2),
        ]

        paths = [
            Path("/test/img1.jpg"),
            Path("/test/img2.jpg"),
            Path("/test/img3.jpg"),
        ]
        poster_paths, classification_data = classifier.filter_posters(paths)

        # Check poster paths
        assert len(poster_paths) == 2
        assert Path("/test/img1.jpg") in poster_paths
        assert Path("/test/img3.jpg") in poster_paths
        assert Path("/test/img2.jpg") not in poster_paths

        # Check classification data for CSV export
        assert len(classification_data) == 3
        assert classification_data[0] == {"filename": "img1.jpg", "classification": "poster"}
        assert classification_data[1] == {"filename": "img2.jpg", "classification": "qr"}
        assert classification_data[2] == {"filename": "img3.jpg", "classification": "poster"}


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_classification_result_creation(self):
        """Test creating a ClassificationResult."""
        result = ClassificationResult(
            image_path=Path("/test/image.jpg"),
            image_type=ImageType.POSTER,
            confidence=1.0,
            raw_response="poster",
            inference_time=1.5,
        )

        assert result.image_path == Path("/test/image.jpg")
        assert result.image_type == ImageType.POSTER
        assert result.confidence == 1.0
        assert result.raw_response == "poster"
        assert result.inference_time == 1.5


class TestImageType:
    """Tests for ImageType enum."""

    def test_image_type_values(self):
        """Test ImageType enum values."""
        assert ImageType.POSTER.value == "poster"
        assert ImageType.QR_CODE.value == "qr"
        assert ImageType.UNKNOWN.value == "unknown"
