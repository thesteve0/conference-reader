"""Tests for CSVExporter class."""

import csv
import tempfile
from pathlib import Path

import pytest

from conference_reader.extraction import ProcessedDocument
from conference_reader.output import CSVExporter


@pytest.fixture
def sample_documents():
    """Create sample ProcessedDocument instances for testing."""
    doc1 = ProcessedDocument(
        filename="poster1.jpg",
        file_path="/data/posters/poster1.jpg",
        extracted_text="# Title One\n\nContent here",
        summary="This is a summary of poster one.",
        success=True,
    )
    doc2 = ProcessedDocument(
        filename="poster2.jpg",
        file_path="/data/posters/poster2.jpg",
        extracted_text="# Title Two\n\nMore content",
        summary="Summary of poster two.",
        success=True,
    )
    doc_failed = ProcessedDocument(
        filename="failed.jpg",
        file_path="/data/posters/failed.jpg",
        extracted_text="",
        success=False,
        error_message="Extraction failed",
    )
    return [doc1, doc2, doc_failed]


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_format_returns_csv_string(self, sample_documents):
        """Test that format() returns valid CSV string."""
        exporter = CSVExporter()
        result = exporter.format(sample_documents)

        assert isinstance(result, str)
        assert "summary,filename,file_path" in result
        assert "poster1.jpg" in result
        assert "poster2.jpg" in result

    def test_format_excludes_failed_documents(self, sample_documents):
        """Test that format() excludes documents with success=False."""
        exporter = CSVExporter()
        result = exporter.format(sample_documents)

        assert "failed.jpg" not in result

    def test_format_row(self, sample_documents):
        """Test that _format_row creates correct dictionary."""
        exporter = CSVExporter()
        row = exporter._format_row(sample_documents[0])

        assert row["summary"] == "This is a summary of poster one."
        assert row["filename"] == "poster1.jpg"
        assert row["file_path"] == "/data/posters/poster1.jpg"

    def test_format_row_handles_none_values(self):
        """Test that _format_row handles None summary."""
        doc = ProcessedDocument(
            filename="test.jpg",
            file_path="/test.jpg",
            extracted_text="content",
            summary=None,
            success=True,
        )
        exporter = CSVExporter()
        row = exporter._format_row(doc)

        assert row["summary"] == ""

    def test_export_creates_file(self, sample_documents):
        """Test that export() creates a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export(sample_documents, "test.csv")

            assert output_path.exists()
            assert output_path.suffix == ".csv"

    def test_export_writes_correct_content(self, sample_documents):
        """Test that export() writes correct CSV content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export(sample_documents, "test.csv")

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have 2 rows (excluding the failed document)
            assert len(rows) == 2
            assert rows[0]["filename"] == "poster1.jpg"
            assert rows[1]["filename"] == "poster2.jpg"

    def test_export_adds_csv_extension(self, sample_documents):
        """Test that export() adds .csv extension if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export(sample_documents, "test")

            assert output_path.suffix == ".csv"
            assert output_path.name == "test.csv"

    def test_export_default_filename(self, sample_documents):
        """Test that export() uses default filename if none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export(sample_documents)

            assert output_path.name == "posters.csv"

    def test_creates_output_directory(self, sample_documents):
        """Test that CSVExporter creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"
            exporter = CSVExporter(output_dir=str(nested_dir))

            assert nested_dir.exists()


class TestCSVExporterClassification:
    """Tests for CSVExporter classification export."""

    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data for testing."""
        return [
            {"filename": "poster1.jpg", "classification": "poster"},
            {"filename": "qr_code.jpg", "classification": "qr"},
            {"filename": "poster2.jpg", "classification": "poster"},
            {"filename": "unknown.jpg", "classification": "unknown"},
        ]

    def test_export_classification_creates_file(self, sample_classification_data):
        """Test that export_classification() creates a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export_classification(sample_classification_data)

            assert output_path.exists()
            assert output_path.suffix == ".csv"

    def test_export_classification_default_filename(self, sample_classification_data):
        """Test that export_classification() uses default filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export_classification(sample_classification_data)

            assert output_path.name == "poster_v_qr.csv"

    def test_export_classification_custom_filename(self, sample_classification_data):
        """Test that export_classification() accepts custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export_classification(
                sample_classification_data, "custom.csv"
            )

            assert output_path.name == "custom.csv"

    def test_export_classification_writes_correct_content(
        self, sample_classification_data
    ):
        """Test that export_classification() writes correct CSV content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export_classification(sample_classification_data)

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 4
            assert rows[0]["filename"] == "poster1.jpg"
            assert rows[0]["classification"] == "poster"
            assert rows[1]["filename"] == "qr_code.jpg"
            assert rows[1]["classification"] == "qr"

    def test_export_classification_headers(self, sample_classification_data):
        """Test that export_classification() uses correct headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            output_path = exporter.export_classification(sample_classification_data)

            with open(output_path, "r") as f:
                first_line = f.readline().strip()

            assert first_line == "filename,classification"
