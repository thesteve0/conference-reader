# Docling Metadata Findings

## Summary
Docling provides rich metadata beyond just extracted text, including confidence scores, quality grades, page dimensions, and structured document elements.

## Key Findings from Exploration

### 1. Confidence Scores ✅
**Available:** YES - This was one of our TODOs to investigate!

The `result.confidence` object provides detailed quality metrics:

```python
result.confidence
├── parse_score: float (nan for images)
├── layout_score: 0.8676 (0-1 scale)
├── table_score: float (nan for images)
├── ocr_score: 0.9547 (0-1 scale)
├── mean_score: 0.9111 (overall quality)
├── low_score: 0.8720 (lowest component score)
├── mean_grade: QualityGrade.EXCELLENT
└── low_grade: QualityGrade.GOOD
```

**Per-Page Confidence:**
- `result.confidence.pages[0]` contains page-specific scores
- Same structure as overall confidence

**Quality Grades Available:**
- EXCELLENT
- GOOD
- (likely FAIR, POOR as well)

### 2. Page Information
**Pages Structure:** Dictionary indexed by page number (1-based)

```python
doc.pages[1]  # First page (PageItem object)
├── page_no: int
├── size: width=3024.0, height=4032.0
└── image: bool (has associated image data)
```

### 3. Processing Status
```python
result.status  # ConversionStatus.SUCCESS
result.errors  # [] (list of any errors)
result.timings # {} (processing time breakdown - empty in our tests)
```

### 4. Document Structure Elements

The `export_to_dict()` provides rich structured data:

```python
doc.export_to_dict()
├── schema_name: str
├── version: str
├── name: str (document name)
├── origin: dict (source metadata)
├── furniture: dict (headers/footers/page numbers)
├── body: dict (main content)
├── groups: dict (content groupings)
├── texts: dict (text elements)
├── pictures: dict (images in document)
├── tables: dict (table structures)
├── key_value_items: dict (form fields/metadata)
├── form_items: dict (form elements)
└── pages: dict (page-level data)
```

### 5. Export Formats Available
- `export_to_markdown()` ✅ (currently using)
- `export_to_dict()` (structured data)
- `export_to_html()` (HTML output)
- `export_to_text()` (plain text)
- `export_to_doctags()` (Docling internal format)
- `export_to_document_tokens()` (tokenized format)
- `export_to_element_tree()` (XML tree structure)

### 6. Document Properties
```python
doc.num_pages  # Total page count
doc.name       # Document name
doc.origin     # Source information
```

## Recommendations for ProcessedDocument Enhancement

Based on these findings, we could add the following fields to ProcessedDocument:

### High Value (Recommended to Add)
1. **Quality Scores:**
   - `quality_score: float` - Overall mean score (0-1)
   - `quality_grade: str` - Mean quality grade ("EXCELLENT", "GOOD", etc.)
   - `ocr_score: float` - OCR-specific confidence (0-1)
   - `layout_score: float` - Layout detection confidence (0-1)

2. **Page Information:**
   - `page_count: int` - Number of pages processed
   - `page_dimensions: dict` - Width/height of each page

### Medium Value (Consider Adding)
3. **Processing Status:**
   - `processing_status: str` - SUCCESS, FAILURE, etc.
   - Already have `success: bool` and `error_message`, but could add explicit status

4. **Structured Elements:**
   - `has_tables: bool` - Whether tables were detected
   - `has_pictures: bool` - Whether images were detected
   - `table_count: int` - Number of tables found
   - `picture_count: int` - Number of images found

### Lower Priority (Probably Not Needed Yet)
5. **Advanced Metadata:**
   - Full structured dict export (very large, probably overkill)
   - Processing timings (empty in our tests anyway)

## Example Usage in Code

### Accessing Confidence Data
```python
result = converter.convert(image_path)

# Get overall quality
mean_score = result.confidence.mean_score  # 0.9111
mean_grade = result.confidence.mean_grade  # QualityGrade.EXCELLENT
ocr_score = result.confidence.ocr_score    # 0.9547
layout_score = result.confidence.layout_score  # 0.8676

# Get page-specific quality
page_confidence = result.confidence.pages[0]
page_quality = page_confidence.mean_grade  # Per-page grade
```

### Accessing Page Data
```python
doc = result.document

# Page count
num_pages = doc.num_pages  # 1

# Page dimensions
first_page = doc.pages[1]  # Dictionary access, 1-based indexing
width = first_page.size.width   # 3024.0
height = first_page.size.height # 4032.0
```

### Accessing Structured Elements
```python
doc_dict = doc.export_to_dict()

tables = doc_dict['tables']      # Table structures
pictures = doc_dict['pictures']  # Image references
has_tables = len(tables) > 0
```

## Next Steps

1. **Decide which fields to add to ProcessedDocument**
   - Quality scores seem most valuable
   - Page information is useful for debugging/reporting

2. **Update ProcessedDocument dataclass**
   - Add new fields with appropriate defaults
   - Update factory methods

3. **Update DocumentExtractor**
   - Extract the new metadata from result object
   - Pass to ProcessedDocument.from_path()

4. **Update main.py display**
   - Show quality scores in output
   - Display page dimensions if useful

5. **Consider future enhancements**
   - Export structured data for downstream processing
   - Filter results by quality threshold
   - Detailed reporting on tables/pictures found
