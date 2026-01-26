"""CSVExporter for writing processed documents to CSV format.

This module exports processed conference poster data to a CSV file
that can be imported into spreadsheets for sharing and review.
"""

import csv
from pathlib import Path
from typing import List

from ..extraction import ProcessedDocument


class CSVExporter:
    """Exports data to CSV format.

    Supports two export types:
    1. ProcessedDocument export (export method):
       - summary: Generated summary
       - filename: Original image filename
       - file_path: Full path to the image

    2. Classification export (export_classification method):
       - filename: Image filename
       - classification: Classification result (poster, qr, unknown)

    Example:
        >>> exporter = CSVExporter()
        >>> exporter.export(documents, Path("output/posters.csv"))
        >>> exporter.export_classification(classification_data)
    """

    HEADERS = ["summary", "filename", "file_path"]
    CLASSIFICATION_HEADERS = ["filename", "classification"]

    def __init__(self, output_dir: str = "output"):
        """Initialize CSVExporter.

        Args:
            output_dir: Directory for output files (created if needed)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_output_path(
        self, output_path: Path | str | None, default_filename: str
    ) -> Path:
        """Resolve and normalize output path.

        Args:
            output_path: User-provided path (absolute, relative, or None)
            default_filename: Default filename if output_path is None

        Returns:
            Resolved absolute Path with .csv extension
        """
        if output_path is None:
            output_file = self.output_dir / default_filename
        else:
            output_file = Path(output_path)
            if not output_file.is_absolute():
                output_file = self.output_dir / output_file

        # Ensure .csv extension
        if output_file.suffix.lower() != ".csv":
            output_file = output_file.with_suffix(".csv")

        return output_file

    def _write_csv(
        self, output_file: Path, headers: list[str], rows: list[dict]
    ) -> Path:
        """Write rows to a CSV file.

        Args:
            output_file: Path to write the CSV file
            headers: List of column headers
            rows: List of dictionaries mapping headers to values

        Returns:
            Path to the written CSV file
        """
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return output_file

    def _format_row(self, doc: ProcessedDocument) -> dict:
        """Format a single document as a CSV row dictionary.

        Args:
            doc: ProcessedDocument to format

        Returns:
            Dictionary mapping column names to values
        """
        return {
            "summary": doc.summary or "",
            "filename": doc.filename,
            "file_path": doc.file_path,
        }

    def format(self, documents: List[ProcessedDocument]) -> str:
        """Format documents as CSV string.

        Args:
            documents: List of ProcessedDocument instances to format

        Returns:
            CSV-formatted string with headers and data rows
        """
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.HEADERS)
        writer.writeheader()

        for doc in documents:
            if doc.success:
                writer.writerow(self._format_row(doc))

        return output.getvalue()

    def export(
        self,
        documents: List[ProcessedDocument],
        output_path: Path | str | None = None,
    ) -> Path:
        """Export documents to a CSV file.

        Args:
            documents: List of ProcessedDocument instances to export
            output_path: Path to output file. If None, uses default name
                in output_dir. If relative, placed in output_dir.

        Returns:
            Path to the written CSV file
        """
        output_file = self._resolve_output_path(output_path, "posters.csv")

        # Filter to successful documents only and format rows
        rows = [self._format_row(doc) for doc in documents if doc.success]

        return self._write_csv(output_file, self.HEADERS, rows)

    def export_classification(
        self,
        results: list[dict[str, str]],
        output_path: Path | str | None = None,
    ) -> Path:
        """Export classification results to a CSV file.

        Args:
            results: List of dicts with 'filename' and 'classification' keys
            output_path: Path to output file. If None, uses 'poster_v_qr.csv'
                in output_dir. If relative, placed in output_dir.

        Returns:
            Path to the written CSV file
        """
        output_file = self._resolve_output_path(output_path, "poster_v_qr.csv")

        return self._write_csv(output_file, self.CLASSIFICATION_HEADERS, results)
