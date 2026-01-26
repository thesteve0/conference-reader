"""Extraction package for extracting text from images using Docling."""

from .document_extractor import DocumentExtractor
from .processed_document import ProcessedDocument

__all__ = ["DocumentExtractor", "ProcessedDocument"]
