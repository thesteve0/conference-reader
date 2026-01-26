"""Image classification module for conference_reader.

This module provides VLM-based classification of conference images
to distinguish full posters from cropped QR code sections.
"""

from .image_classifier import ImageClassifier, ImageType, ClassificationResult
from .vlm_backend import VLMBackend, DeviceMode

__all__ = [
    "ImageClassifier",
    "ImageType",
    "ClassificationResult",
    "VLMBackend",
    "DeviceMode",
]
