"""ImageClassifier for distinguishing posters from QR codes.

This module provides the ImageClassifier class that uses Qwen3-VL-4B
to classify conference images as full posters or cropped QR code sections.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .vlm_backend import VLMBackend, DeviceMode


class ImageType(Enum):
    """Classification result for an image."""

    POSTER = "poster"
    QR_CODE = "qr"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of classifying a single image.

    Attributes:
        image_path: Path to the classified image
        image_type: Classification result (POSTER, QR_CODE, or UNKNOWN)
        confidence: Confidence indicator (1.0 if clear match, 0.5 if unclear)
        raw_response: Full text response from the VLM
        inference_time: Time in seconds for inference
    """

    image_path: Path
    image_type: ImageType
    confidence: float
    raw_response: str
    inference_time: float


class ImageClassifier:
    """Classifies images as posters or QR codes using Qwen3-VL-4B.

    This class provides methods to classify single images or batches,
    and to filter a list of image paths to only those classified as posters.

    Example:
        >>> classifier = ImageClassifier()
        >>> result = classifier.classify(Path("image.jpg"))
        >>> print(result.image_type)
        ImageType.POSTER

        >>> poster_paths = classifier.filter_posters(all_image_paths)
    """

    CLASSIFICATION_PROMPT = """Analyze this image and determine if it shows:
A) A FULL conference poster - You can see the complete poster with:
   - A clear title at the top
   - Multiple sections of content
   - The full layout and structure
   - Sometimes there might be people blocking part of the poster
   - If you can see multiple borders of the poster, it is likely a full poster

B) A CROPPED SECTION - You only see:
   - A zoomed-in portion of a poster
   - Mainly QR codes and very little other content
   - Missing the title and main sections
   - Just a small fragment, not the whole poster

Respond with ONLY ONE WORD:
- Answer "poster" if this is a FULL, complete conference poster
- Answer "qr" if this is a CROPPED section showing mainly QR codes

Your one-word answer:"""

    def __init__(
        self,
        model_name: str = VLMBackend.DEFAULT_MODEL_NAME,
        device_mode: DeviceMode = "eager_float16",
    ):
        """Initialize the classifier with the VLM model.

        Args:
            model_name: HuggingFace model identifier for the VLM
            device_mode: GPU/CPU mode for inference (default: eager_float16)
        """
        self._backend = VLMBackend(
            model_name=model_name,
            device_mode=device_mode,
        )

    def classify(self, image_path: Path) -> ClassificationResult:
        """Classify a single image.

        Args:
            image_path: Path to the image file to classify

        Returns:
            ClassificationResult with the classification and metadata
        """
        # Ensure path is a Path object
        image_path = Path(image_path)

        # Run inference
        raw_response, inference_time = self._backend.generate(
            image_path=image_path,
            prompt=self.CLASSIFICATION_PROMPT,
        )

        # Parse response
        image_type, confidence = self._parse_response(raw_response)

        return ClassificationResult(
            image_path=image_path,
            image_type=image_type,
            confidence=confidence,
            raw_response=raw_response,
            inference_time=inference_time,
        )

    def _parse_response(self, response: str) -> tuple[ImageType, float]:
        """Parse the VLM response to extract classification.

        Args:
            response: Raw text response from the VLM

        Returns:
            Tuple of (ImageType, confidence_score)
        """
        response_lower = response.lower().strip()

        # Check for clear poster classification
        if "poster" in response_lower and "qr" not in response_lower:
            return ImageType.POSTER, 1.0

        # Check for clear QR classification
        if "qr" in response_lower:
            return ImageType.QR_CODE, 1.0

        # Ambiguous response
        return ImageType.UNKNOWN, 0.5

    def classify_batch(
        self, image_paths: list[Path], verbose: bool = False
    ) -> list[ClassificationResult]:
        """Classify multiple images.

        Args:
            image_paths: List of paths to image files
            verbose: If True, print progress during classification

        Returns:
            List of ClassificationResult, one per input image
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            if verbose:
                print(f"Classifying [{i + 1}/{total}]: {path.name}")

            result = self.classify(path)
            results.append(result)

            if verbose:
                print(f"  -> {result.image_type.value} ({result.inference_time:.2f}s)")

        return results

    def _results_to_export_data(
        self, results: list[ClassificationResult]
    ) -> list[dict[str, str]]:
        """Convert ClassificationResults to list of dicts for CSV export.

        Args:
            results: List of ClassificationResult from classify_batch()

        Returns:
            List of dicts with 'filename' and 'classification' keys
        """
        return [
            {
                "filename": result.image_path.name,
                "classification": result.image_type.value,
            }
            for result in results
        ]

    def filter_posters(
        self, image_paths: list[Path], verbose: bool = False
    ) -> tuple[list[Path], list[dict[str, str]]]:
        """Classify images and return poster paths with classification data.

        Args:
            image_paths: List of paths to image files
            verbose: If True, print progress during classification

        Returns:
            Tuple of (poster_paths, classification_data):
            - poster_paths: List of paths classified as posters
            - classification_data: List of dicts with 'filename' and
              'classification' for all images (for CSV export)
        """
        results = self.classify_batch(image_paths, verbose=verbose)

        poster_paths = [
            result.image_path
            for result in results
            if result.image_type == ImageType.POSTER
        ]

        classification_data = self._results_to_export_data(results)

        if verbose:
            print(f"\nFiltered: {len(poster_paths)}/{len(image_paths)} are posters")

        return poster_paths, classification_data

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        self._backend.unload()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False
