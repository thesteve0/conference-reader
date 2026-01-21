# Docling Metrics and Scores - Complete Guide

## Executive Summary

Docling provides confidence metrics to assess document conversion quality. The most important metrics to track are:
- **mean_grade**: Overall quality (EXCELLENT/GOOD/FAIR/POOR)
- **mean_score**: Numerical overall quality (0.0-1.0, higher is better)
- **ocr_score**: OCR accuracy (0.0-1.0)
- **layout_score**: Layout detection quality (0.0-1.0)

**For SmolLM3**: Use `export_to_markdown()` - preserves structure, LLM-friendly, good token efficiency.

---

## Confidence Scores Detailed

### Individual Component Scores

All scores range from **0.0 to 1.0** where higher values indicate better quality.

#### 1. **layout_score**
- **What it measures**: Overall quality of document element recognition
- **Range**: 0.0 to 1.0
- **Example**: 0.8676 (86.76% confidence in layout detection)
- **Meaning**: How well Docling identified structural elements (headings, paragraphs, lists, tables, etc.)
- **Important for**: Documents with complex layouts, multi-column formats

#### 2. **ocr_score**
- **What it measures**: Quality of OCR-extracted content
- **Range**: 0.0 to 1.0
- **Example**: 0.9547 (95.47% confidence in OCR)
- **Meaning**: How accurately text was extracted from images
- **Important for**: All image-based documents (conference posters, scanned PDFs)

#### 3. **parse_score**
- **What it measures**: 10th percentile score of digital text cells (emphasizes problem areas)
- **Range**: 0.0 to 1.0
- **Example**: nan (not applicable for image-only documents)
- **Meaning**: Quality of parsing digitally-embedded text (not OCR'd text)
- **Important for**: PDFs with embedded text (not relevant for our image posters)
- **Note**: Will be `nan` for pure image files

#### 4. **table_score**
- **What it measures**: Table extraction quality
- **Range**: 0.0 to 1.0
- **Example**: nan (not yet implemented by Docling)
- **Status**: **NOT YET IMPLEMENTED** - always returns `nan`
- **Important for**: Future feature for documents with complex tables

### Summary Scores

#### 5. **mean_score**
- **What it measures**: Average of all component scores
- **Calculation**: Mean of parse_score, layout_score, table_score, ocr_score (ignoring NaN values)
- **Range**: 0.0 to 1.0
- **Example**: 0.9111 (91.11% overall quality)
- **Use case**: Primary numerical metric for overall conversion quality

#### 6. **low_score**
- **What it measures**: 5th percentile score (highlights worst-performing areas)
- **Calculation**: 5th percentile of component scores
- **Range**: 0.0 to 1.0
- **Example**: 0.8720
- **Use case**: Identifies quality floor - useful for quality control

### Quality Grades (Categorical)

#### 7. **mean_grade**
- **What it measures**: Overall categorical quality rating
- **Values**:
  - `QualityGrade.EXCELLENT`
  - `QualityGrade.GOOD`
  - `QualityGrade.FAIR`
  - `QualityGrade.POOR`
- **Example**: `QualityGrade.EXCELLENT`
- **Use case**: User-friendly quality assessment, filtering/reporting

#### 8. **low_grade**
- **What it measures**: Quality grade for worst-performing areas
- **Values**: EXCELLENT / GOOD / FAIR / POOR
- **Example**: `QualityGrade.GOOD`
- **Use case**: Conservative quality threshold (if worst areas are GOOD, entire doc is solid)

### Per-Page Confidence

All confidence metrics are available at the page level:
```python
result.confidence.pages[0]  # Page 0 (0-indexed)
├── parse_score: nan
├── layout_score: 0.8676
├── table_score: nan
├── ocr_score: 0.9547
├── mean_score: 0.9111
├── low_score: 0.8720
├── mean_grade: EXCELLENT
└── low_grade: GOOD
```

**Use case**: Multi-page documents where quality varies by page.

---

## Processing Time (Timings)

### result.timings
- **Type**: `dict`
- **Current value**: `{}` (empty)
- **Status**: Not populated in current Docling version
- **Alternative**: Check Docling logs for timing info
  - Example from logs: "Finished converting document good-test.jpg in 7.31 sec."
  - **Recommendation**: Parse processing time from logs if needed, or measure externally

### result.timestamp
- **Type**: `None` (not set)
- **Status**: Not populated in current version

**Note**: For our use case, we can track processing time ourselves in ProcessedDocument using `extraction_time` field we already have.

---

## Export Formats for SmolLM3

### Comparison

| Format | Size (chars) | Pros | Cons | Recommendation |
|--------|-------------|------|------|----------------|
| **Markdown** | 2,437 | ✓ Structure preserved<br>✓ LLM-trained on markdown<br>✓ Good readability | Slightly larger than text | **✅ RECOMMENDED** |
| **Text** | 2,357 | ✓ Smallest size<br>✓ Plain text | ✗ Loses structure<br>✗ No headings/lists | Use if tokens critical |
| **Dict** | N/A | ✓ Full structured data<br>✓ Programmatic access | ✗ Complex structure<br>✗ Needs processing | Not for LLM input |
| **HTML** | 5,183 | ✓ Rich formatting | ✗ Verbose (2x markdown)<br>✗ May confuse LLMs | ❌ Not recommended |

### Markdown Example
```markdown
## CURE-Bench NeurIPs 2025

<!-- image -->

## CliniTHink: Think Like a Clinician Multi-Agent Therapeutic Reasoning...

AmrestChinkamolNatpatharaongjirapatrittaphasChaisutyakornNaphatornwichaiuawit...
```

### Text Example
```
## CURE-Bench NeurIPs 2025

## CliniTHink: Think Like a Clinician Multi-Agent Therapeutic Reasoning...

AmrestChinkamolNatpatharaongjirapatrittaphasChaisutyakornNaphatornwichaiuawit...
```

**Key Difference**: Markdown preserves `<!-- image -->` markers and formatting cues that help LLMs understand document structure.

### **Decision: Use `export_to_markdown()` for SmolLM3**

Reasons:
1. Most LLMs (including SmolLM3) are trained on markdown-formatted text
2. Preserves document structure (headings, lists, emphasis)
3. Only ~3% larger than plain text (80 chars)
4. Better prompt context for summarization

---

## Best Practices from Docling Documentation

From the [official Docling documentation](https://docling-project.github.io/docling/concepts/confidence_scores/):

1. **Focus on grades, not raw scores**: Use `mean_grade` and `low_grade` for assessment
2. **Scores are for internal use**: Raw numerical scores may change in future versions
3. **Introduced in v2.34.0**: Confidence scoring is a relatively new feature

### Quality Thresholds (Recommended)

For automated processing workflows:

| Grade | Use Case |
|-------|----------|
| **EXCELLENT** | ✓ Safe to use without manual review |
| **GOOD** | ✓ Generally reliable, spot-check if critical |
| **FAIR** | ⚠️ Review recommended, may have issues |
| **POOR** | ❌ Manual review required, significant quality issues |

For conference poster processing:
- Accept: EXCELLENT, GOOD
- Review: FAIR
- Reject or flag: POOR

---

## What to Add to ProcessedDocument

### Recommended Fields (High Priority)

```python
@dataclass
class ProcessedDocument:
    # Existing fields
    filename: str
    file_path: str
    extracted_text: str
    extraction_time: datetime
    success: bool
    error_message: Optional[str]

    # NEW: Quality metrics
    quality_grade: Optional[str] = None          # "EXCELLENT", "GOOD", "FAIR", "POOR"
    quality_score: Optional[float] = None        # 0.0-1.0 (mean_score)
    ocr_score: Optional[float] = None            # 0.0-1.0
    layout_score: Optional[float] = None         # 0.0-1.0
    low_quality_grade: Optional[str] = None      # Conservative quality estimate
```

### Lower Priority (Consider Later)

```python
    # Page information
    page_count: Optional[int] = None

    # Structured elements (if needed for filtering/reporting)
    table_count: Optional[int] = None
    picture_count: Optional[int] = None
```

### Not Recommended

- `timings` - empty in current version, we already track extraction_time
- `parse_score` - always nan for images
- `table_score` - not implemented yet
- Full dict export - too large, not needed

---

## Code Examples

### Accessing Confidence Data

```python
result = converter.convert(image_path)

# Get overall quality (RECOMMENDED)
mean_grade = str(result.confidence.mean_grade)     # "EXCELLENT"
low_grade = str(result.confidence.low_grade)       # "GOOD"

# Get numerical scores (if needed)
mean_score = result.confidence.mean_score          # 0.9111
ocr_score = result.confidence.ocr_score            # 0.9547
layout_score = result.confidence.layout_score      # 0.8676

# Create ProcessedDocument with quality metrics
doc = ProcessedDocument.from_path(
    file_path=image_path,
    extracted_text=result.document.export_to_markdown(),  # ← Use markdown for SmolLM3
    quality_grade=str(result.confidence.mean_grade),
    quality_score=result.confidence.mean_score,
    ocr_score=result.confidence.ocr_score,
    layout_score=result.confidence.layout_score,
    low_quality_grade=str(result.confidence.low_grade)
)
```

### Filtering by Quality

```python
# Filter out low-quality extractions
high_quality_docs = [
    doc for doc in documents
    if doc.quality_grade in ["EXCELLENT", "GOOD"]
]

# Conservative filter (using low_grade)
very_high_quality_docs = [
    doc for doc in documents
    if doc.low_quality_grade in ["EXCELLENT", "GOOD"]
]
```

---

## Sources

- [Docling Confidence Scores Documentation](https://docling-project.github.io/docling/concepts/confidence_scores/)
- Docling v2.34.0+ (confidence grades feature)

---

## Next Steps

1. ✅ Update ProcessedDocument dataclass with quality fields
2. ✅ Update DocumentExtractor to extract confidence metrics
3. ✅ Confirm using `export_to_markdown()` for SmolLM3 integration
4. ✅ Update main.py to display quality metrics in output
5. Consider quality-based filtering or warnings in the pipeline
