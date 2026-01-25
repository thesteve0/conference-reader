# Next Session: Conference Poster Processing Pipeline

**Created**: 2026-01-25
**Status**: Ready to implement

---

## Goal

Build an end-to-end pipeline that processes conference poster images into a structured CSV spreadsheet, using proper library classes (not scripts).

---

## Pipeline Overview

```
[Raw Images]
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Image Classification   │
│ ImageClassifier class           │
│ Filter: poster vs QR code       │
└─────────────────────────────────┘
    │
    │ (posters only)
    ▼
┌─────────────────────────────────┐
│ Stage 2: Text Extraction        │
│ TextExtractor class             │
│ Create processed documents      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 3: Text Summarization     │
│ TextSummarizer class            │
│ Extract key metadata            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 4: CSV Export             │
│ CSVExporter class               │
│ Output structured spreadsheet   │
└─────────────────────────────────┘
```

---

## Architecture

### Library Structure

```
conference_reader/
├── __init__.py
├── classifier/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   ├── image_classifier.py  # ImageClassifier - poster vs QR
│   └── vlm_backend.py       # Qwen3-VL-4B backend wrapper
├── extractor/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   └── text_extractor.py    # TextExtractor class
├── summarizer/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   └── text_summarizer.py   # TextSummarizer class
├── exporter/
│   ├── __init__.py
│   └── csv_exporter.py      # CSVExporter class
├── models/
│   ├── __init__.py
│   ├── document.py          # ProcessedDocument dataclass
│   └── poster_metadata.py   # PosterMetadata dataclass
└── config/
    ├── __init__.py
    └── rocm_config.py       # ROCm stability settings
```

---

## Stage 1: Image Classifier (First to Implement)

### Class Design

```python
# conference_reader/classifier/image_classifier.py

from pathlib import Path
from enum import Enum
from dataclasses import dataclass

class ImageType(Enum):
    POSTER = "poster"
    QR_CODE = "qr"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    image_path: Path
    image_type: ImageType
    confidence: float
    raw_response: str
    inference_time: float

class ImageClassifier:
    """Classifies images as posters or QR codes using Qwen3-VL-4B."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):
        """Initialize the classifier with the VLM model."""
        pass

    def classify(self, image_path: Path) -> ClassificationResult:
        """Classify a single image."""
        pass

    def classify_batch(self, image_paths: list[Path]) -> list[ClassificationResult]:
        """Classify multiple images."""
        pass

    def filter_posters(self, image_paths: list[Path]) -> list[Path]:
        """Return only paths classified as posters."""
        pass
```

### VLM Backend Wrapper

```python
# conference_reader/classifier/vlm_backend.py

class VLMBackend:
    """Wrapper for Qwen3-VL-4B with ROCm stability settings."""

    def __init__(
        self,
        model_name: str,
        device_mode: str = "eager_float16",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
    ):
        """Load model with proper ROCm configuration."""
        pass

    def generate(self, image_path: Path, prompt: str) -> str:
        """Run inference on an image with a prompt."""
        pass

    def unload(self):
        """Free GPU memory."""
        pass
```

### ROCm Configuration

```python
# conference_reader/config/rocm_config.py

import os

def apply_rocm_stability_settings():
    """Apply environment variables for stable ROCm execution."""
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["HSA_ENABLE_INTERRUPT"] = "0"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "tunableop_results.csv"
```

---

## Data Models

```python
# conference_reader/models/document.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessedDocument:
    """Represents extracted text from a poster image."""
    source_image: Path
    raw_text: str
    sections: dict[str, str]  # e.g., {"title": "...", "abstract": "..."}
    extraction_time: float

# conference_reader/models/poster_metadata.py

@dataclass
class PosterMetadata:
    """Structured metadata extracted from a poster."""
    filename: str
    title: str
    authors: list[str]
    institution: str
    summary: str
    keywords: list[str]
    conference: str | None
    year: int | None
```

---

## Implementation Order

### Session 1: Image Classifier
1. Create package structure (`conference_reader/`)
2. Implement `ROCmConfig` (extract settings from run_stable.sh)
3. Implement `VLMBackend` (refactor from experiment_qwen_vl.py)
4. Implement `ImageClassifier` with `classify()` and `filter_posters()`
5. Write tests

### Session 2: Text Extractor
1. Implement `TextExtractor` class
2. Decide: VLM-based or OCR-based extraction
3. Define `ProcessedDocument` model

### Session 3: Text Summarizer
1. Implement `TextSummarizer` class
2. Define `PosterMetadata` model
3. Create extraction prompts

### Session 4: CSV Exporter & Pipeline
1. Implement `CSVExporter` class
2. Create main pipeline orchestrator
3. End-to-end testing

---

## Refactoring Notes

### Code to Extract from experiment_qwen_vl.py

| Current Location | New Location |
|------------------|--------------|
| Model loading logic | `VLMBackend.__init__()` |
| `classify_image()` function | `ImageClassifier.classify()` |
| `CLASSIFICATION_PROMPT` | `ImageClassifier._prompt` |
| Device mode handling | `VLMBackend` constructor |
| Processor configuration | `VLMBackend` (min/max pixels) |

### Keep experiment_qwen_vl.py As-Is
- Keep the script for standalone testing/debugging
- Library classes will contain the production code

---

## Technical Requirements

### ROCm Stability (Mandatory)
All VLM operations must use:
- `eager_float16` precision (NOT bfloat16)
- Reduced image resolution (MAX_PIXELS = 512 * 28 * 28)
- Stability environment variables (via `apply_rocm_stability_settings()`)
- TunableOp cached kernels

### Memory Management
- Load model once, reuse for batch operations
- Provide `unload()` method to free GPU memory
- Consider context manager pattern for automatic cleanup

---

## Usage Example (Target API)

```python
from conference_reader.classifier import ImageClassifier
from conference_reader.extractor import TextExtractor
from conference_reader.summarizer import TextSummarizer
from conference_reader.exporter import CSVExporter
from conference_reader.config import apply_rocm_stability_settings
from pathlib import Path

# Apply ROCm settings before any GPU operations
apply_rocm_stability_settings()

# Stage 1: Classify and filter
classifier = ImageClassifier()
image_dir = Path("/data/neurips/poster_images")
all_images = list(image_dir.glob("*.jpg"))
poster_images = classifier.filter_posters(all_images)

# Stage 2: Extract text
extractor = TextExtractor()
documents = [extractor.extract(img) for img in poster_images]

# Stage 3: Summarize
summarizer = TextSummarizer()
metadata = [summarizer.summarize(doc) for doc in documents]

# Stage 4: Export
exporter = CSVExporter()
exporter.export(metadata, output_path=Path("output/posters.csv"))
```

---

## Ready to Start

When you begin next session, say:
> "Let's create the conference_reader library starting with the ImageClassifier class"

We'll start by creating the package structure and implementing Stage 1.
