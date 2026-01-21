"""ImageLoader class for discovering and listing image files."""

from pathlib import Path
from typing import List


class ImageLoader:
    """Discovers and loads image file paths from a directory.

    This class scans a directory for JPG/JPEG image files and returns
    their absolute paths for processing.

    Attributes:
        directory: Path object pointing to the directory containing images
        VALID_EXTENSIONS: Set of valid image file extensions (case-insensitive)
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.JPG', '.JPEG'}

    def __init__(self, directory: str):
        """Initialize the ImageLoader with a directory path.

        Args:
            directory: Path to directory containing image files

        Raises:
            FileNotFoundError: If the directory does not exist
            ValueError: If the path exists but is not a directory
        """
        self.directory = Path(directory)

        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

    def get_image_paths(self) -> List[str]:
        """Get sorted list of absolute paths to JPG/JPEG files in the directory.

        Returns:
            Sorted list of absolute file paths as strings

        Raises:
            ValueError: If no valid image files are found in the directory
        """
        image_paths = []

        # Iterate through all files in the directory
        for file_path in self.directory.iterdir():
            if file_path.is_file() and self._is_valid_image(file_path):
                image_paths.append(str(file_path.absolute()))

        if not image_paths:
            raise ValueError(f"No valid image files found in: {self.directory}")

        # Return sorted list for consistent ordering
        return sorted(image_paths)

    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if a file has a valid image extension.

        Args:
            file_path: Path object to check

        Returns:
            True if file has .jpg or .jpeg extension (case-insensitive)
        """
        return file_path.suffix in self.VALID_EXTENSIONS
