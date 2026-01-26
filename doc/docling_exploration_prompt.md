# Docling OCR Engine & Configuration Exploration

## Context & Background

**Project:** conference-reader - Extract and summarize text from conference poster images
**Current Issue:** Poster validation accuracy is suboptimal (36-73% depending on thresholds)
**Key Problem:** Some valid posters fail OCR extraction (e.g., IMG_1222 extracts only 30 chars)

## Current Setup

**Docling Usage:**
- Currently using **default configuration** (auto-selected OCR engine)
- Auto-selected engine: **RapidOCR with torch backend**
- GPU: AMD ROCm (Framework 13 laptop)
- Environment: Python 3.13, devcontainer with ROCm base image

**Current Code Location:**
- DocumentExtractor: `src/conference_reader/extraction/document_extractor.py:27`
  ```python
  self.converter = DocumentConverter()  # Default config
  ```

## Available OCR Engines

According to Docling logs, supported engines include:
- `auto` (current selection → RapidOCR)
- `rapidocr` (selected by default)
- `easyocr`
- `tesserocr`
- `tesseract`
- `ocrmac`

**Currently unavailable** (would need installation):
- `easyocr` - "not installed"
- RapidOCR with `onnxruntime` - "not installed"

## Problem Cases

### IMG_1222.JPEG - Severe OCR Failure
- **Extracted:** 30 chars ("<!-- image -->")
- **Expected:** Valid poster (labeled true in ground truth)
- **Likely cause:** Image-heavy poster with minimal text extraction
- **Question:** Would different OCR engine extract more text?

### IMG_1199.JPG - Valid but Only 1 Heading
- **Extracted:** 1926 chars, 1 heading, 98 children
- **Issue:** Fails heading_count threshold but is valid
- **Question:** Could better extraction improve heading detection?

### IMG_1227.JPEG - Valid but Low Structure
- **Extracted:** 1006 chars, 1 heading, 13 children
- **Issue:** Low children count for a valid poster
- **Question:** Is layout detection missing elements?

## Research Questions

### 1. OCR Engine Comparison
**Goal:** Determine if different OCR engines extract more/better text

**Test cases to evaluate:**
1. IMG_1222.JPEG (current: 30 chars) - Does tesseract/easyocr extract more?
2. All 11 images - Which engine produces highest character counts?
3. Quality scores - Do different engines affect confidence scores?

**Specific investigation:**
- Compare text extraction length across engines
- Check if heading detection improves
- Evaluate impact on children_count (document structure)

### 2. Docling Configuration Options
**Goal:** Understand what configuration options are available beyond OCR engine

**Areas to explore:**
- **Image preprocessing:** Can we adjust contrast, resolution, or other parameters?
- **Layout detection models:** Are there alternative layout detection models?
- **Table structure:** Settings for `docling_tableformer` (if relevant to posters)
- **Pipeline options:** What does `StandardPdfPipeline` offer for images?
- **Quality thresholds:** Can we configure what Docling considers "good" quality?

### 3. Best Practices for Poster Images
**Goal:** Learn optimal Docling configuration for academic poster images

**Considerations:**
- Posters often have:
  - Complex multi-column layouts
  - Mixed text sizes (titles, headers, body)
  - Embedded figures and diagrams
  - Varied fonts and styling

**Questions:**
- Are there poster-specific best practices?
- Should we use different settings than for PDF documents?
- How to balance accuracy vs processing time?

## Installation Considerations

**Current environment:** Devcontainer with ROCm, uv package manager

**If recommending new OCR engines:**
1. Check ROCm compatibility (AMD GPU, not NVIDIA)
2. Verify Python 3.13 support
3. Consider installation complexity (prefer pip/uv installable packages)
4. Evaluate additional dependencies (e.g., tesseract system packages)

**Package management:**
- Primary: `uv add <package>` (preferred)
- Fallback: `pip install <package>` (if uv incompatible)
- System: `apt-get install` (if needed for tesseract, etc.)

## Desired Output from Exploration

### 1. Concrete Recommendations
- **Best OCR engine** for our use case
- **Recommended Docling configuration** (with code example)
- **Installation steps** if new packages needed

### 2. Comparative Analysis
- **Performance comparison** across OCR engines for our 11 test images
- **Table format:**
  ```
  | Filename      | RapidOCR | Tesseract | EasyOCR | Recommendation |
  |---------------|----------|-----------|---------|----------------|
  | IMG_1222.JPEG | 30 chars | XXX chars | XXX chars | <best engine> |
  ```

### 3. Implementation Guidance
- **Code changes** needed in DocumentExtractor
- **Configuration parameters** to expose (if any)
- **Trade-offs** (accuracy vs speed, memory usage, etc.)

### 4. Validation Impact Assessment
- **Expected improvement** in validation accuracy
- **Which false negatives** would likely be fixed
- **Risk assessment** (could changes make things worse?)

## Testing Approach

### Suggested Workflow
1. **Quick diagnostic test** on IMG_1222 with 2-3 different engines
2. **If promising**, run full comparison on all 11 images
3. **Measure impact** on our validation metrics (text_length, heading_count, children_count)
4. **Report findings** with recommendation

### Success Criteria
- IMG_1222 extracts >200 chars (vs current 30)
- Overall validation accuracy improves >10 percentage points
- No degradation in currently passing images
- Installation/configuration is straightforward

## Reference Files

**Key codebase locations:**
- `src/conference_reader/extraction/document_extractor.py` - DocumentExtractor class
- `src/conference_reader/classifier/image_classifier.py` - VLM-based image classification
- `datasets/eval/validator_ground_truth.json` - Ground truth labels (11 images)

**Note:** Heuristic validation (valid_image_config.py) has been replaced with
VLM-based classification using Qwen3-VL-4B. The ImageClassifier now handles
poster vs QR code filtering before text extraction.

**Test data:**
- Directory: `/data/neurips/invalid_poster_images/`
- 11 labeled images (4 valid, 7 invalid)
- Representative mix of poster types and quality levels

## Additional Context

**Current classification approach:**
- VLM-based classification using Qwen3-VL-4B
- ImageClassifier filters posters vs QR codes before text extraction
- Heuristic validation has been removed in favor of VLM classification

**Why OCR quality still matters:**
- Better OCR → more text extracted → better summaries
- Better layout detection → improved title extraction
- Overall: Better extraction improves the quality of the final output

---

## How to Use This Prompt

**Start new Claude Code session:**
1. Copy this entire file content
2. Paste into new Claude Code session
3. Claude will have full context to explore Docling options
4. Claude can test different engines, compare results, and provide recommendations

**Expected session outcome:**
- Specific recommendation (e.g., "Switch to Tesseract OCR")
- Code changes needed
- Installation instructions
- Comparative data showing improvement