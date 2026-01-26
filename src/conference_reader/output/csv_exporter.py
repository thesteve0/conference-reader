"""CSVExporter for writing processed documents to CSV format.

This module exports processed conference poster data to a CSV file
that can be imported into spreadsheets for sharing and review.
"""

import csv
from pathlib import Path
from typing import List

from ..extraction import ProcessedDocument


class CSVExporter:
    """Exports ProcessedDocument data to CSV format.

    The CSV output includes:
    - title: Poster title (from ProcessedDocument.title)
    - summary: Generated summary (from ProcessedDocument.summary)
    - filename: Original image filename
    - file_path: Full path to the image

    Example:
        >>> exporter = CSVExporter()
        >>> exporter.export(documents, Path("output/posters.csv"))
    """

    HEADERS = ["title", "summary", "filename", "file_path"]

    def __init__(self, output_dir: str = "output"):
        """Initialize CSVExporter.

        Args:
            output_dir: Directory for output files (created if needed)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_row(self, doc: ProcessedDocument) -> dict:
        """Format a single document as a CSV row dictionary.

        Args:
            doc: ProcessedDocument to format

        Returns:
            Dictionary mapping column names to values
        """
        return {
            "title": doc.title or "",
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
        # Determine output path
        if output_path is None:
            output_file = self.output_dir / "posters.csv"
        else:
            output_file = Path(output_path)
            if not output_file.is_absolute():
                output_file = self.output_dir / output_file

        # Ensure .csv extension
        if output_file.suffix.lower() != ".csv":
            output_file = output_file.with_suffix(".csv")

        # Filter to successful documents only
        successful_docs = [doc for doc in documents if doc.success]

        # Write CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()

            for doc in successful_docs:
                writer.writerow(self._format_row(doc))

        return output_file
