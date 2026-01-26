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
        title="Title One",
        summary="This is a summary of poster one.",
        success=True,
    )
    doc2 = ProcessedDocument(
        filename="poster2.jpg",
        file_path="/data/posters/poster2.jpg",
        extracted_text="# Title Two\n\nMore content",
        title="Title Two",
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
        assert "title,summary,filename,file_path" in result
        assert "Title One" in result
        assert "Title Two" in result

    def test_format_excludes_failed_documents(self, sample_documents):
        """Test that format() excludes documents with success=False."""
        exporter = CSVExporter()
        result = exporter.format(sample_documents)

        assert "failed.jpg" not in result

    def test_format_row(self, sample_documents):
        """Test that _format_row creates correct dictionary."""
        exporter = CSVExporter()
        row = exporter._format_row(sample_documents[0])

        assert row["title"] == "Title One"
        assert row["summary"] == "This is a summary of poster one."
        assert row["filename"] == "poster1.jpg"
        assert row["file_path"] == "/data/posters/poster1.jpg"

    def test_format_row_handles_none_values(self):
        """Test that _format_row handles None title/summary."""
        doc = ProcessedDocument(
            filename="test.jpg",
            file_path="/test.jpg",
            extracted_text="content",
            title=None,
            summary=None,
            success=True,
        )
        exporter = CSVExporter()
        row = exporter._format_row(doc)

        assert row["title"] == ""
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
            assert rows[0]["title"] == "Title One"
            assert rows[1]["title"] == "Title Two"

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
