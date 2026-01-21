"""Configuration for poster image validation."""

from dataclasses import dataclass


@dataclass
class ValidImageConfig:
    """Configuration for poster image validation.

    This class defines thresholds for determining whether an extracted
    image represents a valid conference poster or should be filtered out
    (e.g., QR codes, partial images, low-quality scans).

    Attributes:
        min_text_length: Minimum characters for valid poster. Images with
            less extracted text are considered invalid. Default: 800.
        require_heading: Whether to require a markdown heading (# or ##)
            near the beginning of the extracted text. Valid posters
            typically have titles formatted as headings; QR codes and
            invalid images don't. Default: True.
        min_heading_count: Minimum number of markdown headings (# or ##)
            in the extracted text. Valid posters have multiple sections;
            partial images and QR code crops typically have fewer headings.
            Default: 3.
        min_children_count: Minimum number of document structure children
            elements. Valid posters have rich document structure (50+
            children); partial images and code snippets have simpler
            structure (< 20 children). Default: 50.

    Validation Logic:
        An image is marked as INVALID only if it fails ALL checks (AND
        logic). This permissive approach means an image passes if it meets
        ANY single threshold, reducing false negatives while still catching
        clearly invalid images like QR codes.

    Example:
        >>> config = ValidImageConfig(
        ...     min_text_length=1000, min_heading_count=4
        ... )
        >>> # Use stricter validation for higher quality requirements
        >>> config2 = ValidImageConfig(
        ...     require_heading=False, min_heading_count=0
        ... )
        >>> # Disable heading requirements if needed
    """

    min_text_length: int = 800
    require_heading: bool = True
    min_heading_count: int = 3
    min_children_count: int = 50
