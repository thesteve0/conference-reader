"""Output writing for conference reader results (to be implemented)."""

from pathlib import Path
from typing import List

from ..extraction import ProcessedDocument


class OutputWriter:
    """Writes output in markdown or CSV format.

    TODO: Implement output generation once summarization is working.
    """

    DEFAULT_OUTPUT_DIR = "output"

    def __init__(self, format: str, output_path: str = "conference_results"):
        """Initialize the OutputWriter.

        Args:
            format: Output format, either "csv" or "md"
            output_path: Base name for output file (extension added automatically)

        Raises:
            ValueError: If format is not "csv" or "md"
            NotImplementedError: This class is not yet implemented
        """
        if format not in ("csv", "md"):
            raise ValueError(f"Invalid format: {format}. Must be 'csv' or 'md'")

        self.format = format

        # Create output directory if it doesn't exist
        output_dir = Path(self.DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # Build full path with automatic extension
        filename = output_path if output_path.endswith(f".{format}") else f"{output_path}.{format}"
        self.output_path = output_dir / filename

        raise NotImplementedError("OutputWriter not yet implemented")

    def write_output(self, documents: List[ProcessedDocument]) -> None:
        """Format and write documents to output file.

        This method handles both formatting and writing in a single call.

        Args:
            documents: List of ProcessedDocument instances to output

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("OutputWriter not yet implemented")

    def _format_results(self, documents: List[ProcessedDocument]) -> str:
        """Private method to format documents (used internally by write_output).

        Args:
            documents: List of ProcessedDocument instances to format

        Returns:
            Formatted output string

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("OutputWriter not yet implemented")
