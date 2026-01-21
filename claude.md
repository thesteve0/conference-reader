## project overview

This project is intended to be run against directories of images taken at a conference or a meetup. The idea is to provide summarized output of the 
images to help non-attendees decide which images they want to look at. Primary focus is on images of posters since those are self-contained

### Key objectives

1. Output a list of images and a brief summary allowing non-attendees to quickly scan and decide which images they want to open
2. Learn how to use Docling
3. Learn how to use SmolLLM to summarize the text
4. Write a cleanly and logically organized code base that follows best practices in:
  a. Object Oriented Programming
  b. Python
  c. Testing
  d. AI model calling and chaining 

## architecture decisions

We are going to start small but try to work through the entire process with a few images. We will use the simplest model to implement and run. Their implementation will done in a way that will make it straightfoward to swap in new models or change important parameters.  Once the whole flow working with passing test then we will refactor by testing alternative models and make it scalable. 
We are going to add functionality in the following order:
1. Image reading and text extraction from the test directory
2. Then we will add the piece to summarize the extracted text
3. Finally, we will add the piece to output the CSV file

1. We are using Docling for the image text and figure extraction
https://www.docling.ai/
https://docling-project.github.io/docling/
https://github.com/docling-project/docling

2. We use SMolLM for text summarization
https://huggingface.co/collections/HuggingFaceTB/smollm3
https://huggingface.co/blog/smollm3
https://huggingface.co/HuggingFaceTB/SmolLM3-3B
https://github.com/huggingface/smollm


### Core Components

The GPU accelerator we will be using for this work will be an AMD Stix Halo chipset with at least 48 gigs of RAM to run the models. The chips are:
1. AMD RYZEN AI MAX+ 395 with 128 Gigs of RAM
2. AMD RYZEN AI 9 HX Pro 370 with 96 gigs of RAM

We will be working in a devcontainer that is based off of the AMD PyTorch container so ROCm acceleration is available for all our work. It is ROCm 7.1 with PyTorch 2.9.1
`rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1`

## codebase structure

my-ml-project/
├── .devcontainer/
│   ├── devcontainer.json      # VSCode devcontainer configuration
│   └── ...                    # Additional VSCode-specific files
├── .idea/                     # JetBrains configuration (if selected)
├── configs/                   # Configuration files
├── doc/                       # Documentation
│   └── issues.md              # Issue tracking and improvements (active collaboration)
├── scripts/                   # Utility scripts
├── src/                       # Source code
├── tests/                     # Test files
├── models/                    # Model storage (volume mount)
├── datasets/                  # Dataset storage (volume mount)
└── .cache/                    # Cache directory (volume mount)

## Input/Output Specifications

### Input
- Supported image formats: JPG or HEIC. 
- Directory structure expectations
- Image naming conventions (if any)

### Output
- Format: CSV which will be imported into a spreadsheet. Since this will be shared with others in Red Hat it would be good if we can create google drive links to the actual images
- Fields in the summary: Extracted image title, summary, link to image, filename, confidence scores
- Example output structure "CrypticBio: A large multimodal dataset for visually confusing species", "A dataset was created and test for working on building and testing models ability to detect differences in similar species. Comparisons of different model results are given", "posters_test/image01.jpg", "image01.jpg", 0.56

## Testing Approach

- Unit tests: Individual components (Docling wrapper, SmolLM interface)
- Integration tests: End-to-end pipeline
- Test data: Sample conference images are in /data/neurips/poster_test

## critical patterns 

## Dependency Management Philosophy

### The Problem

ROCm containers (like NVIDIA containers) come with pre-installed optimized libraries. Installing packages from PyPI that conflict with these can break GPU support or introduce version conflicts.

### The Solution

The `resolve-dependencies.py` script:
- Reads `requirements.txt` or `pyproject.toml`
- Compares against ROCm-provided packages
- Creates filtered versions that skip conflicting packages
- Installs remaining dependencies using `uv` into the system environment

This preserves ROCm optimizations while allowing additional package installation.

### Virtual Environment Design - CRITICAL: Python Version Matching

The template uses a `.pth` bridge file approach (NOT `--system-site-packages`) to make container packages accessible while preventing accidental overwrites.

**Why .pth instead of --system-site-packages:**
- `--system-site-packages` would allow `pip install torch` to overwrite ROCm packages, even with uv's `exclude-dependencies`
- The `.pth` file makes packages importable but doesn't affect pip's package resolution
- This provides stronger protection against accidental overwrites via direct pip usage

**CRITICAL REQUIREMENT: Python Version Must Match**

The `.venv` MUST be created with `/opt/venv/bin/python` to ensure Python version consistency:

- Container's `/opt/venv` uses Python 3.13 (as of ROCm 7.1 containers)
- If `.venv` is created with system Python 3.12, **binary incompatibility** breaks numpy/torch imports
- The misleading error "importing numpy from source directory" actually means "C extension binary incompatibility"
- Python 3.12 cannot load `.so` files compiled for Python 3.13 (and vice versa)

**How It Works:**

1. `setup-environment.sh` detects container Python version: `/opt/venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
2. Creates `.venv` using that Python: `/opt/venv/bin/python -m venv .venv`
3. Verifies versions match after creation (exits with error if mismatch detected)
4. Creates dynamic `.pth` bridge: `.venv/lib/pythonX.Y/site-packages/_rocm_bridge.pth` → `/opt/venv/lib/pythonX.Y/site-packages`
5. Python loads the .pth file and adds `/opt/venv` to sys.path
6. Container packages (torch, numpy) become importable
7. uv's `exclude-dependencies` still prevents installing excluded packages

**Version Mismatch Detection:**

The template now automatically verifies Python versions match during venv creation and will error with a clear message if they don't. This prevents the silent failure mode that causes confusing import errors later.

**Common Scenario That Triggers Mismatch:**

- Manually running `python3 -m venv .venv` instead of `/opt/venv/bin/python -m venv .venv`
- The system `python3` might be a different version than the container's Python
- This creates a venv with the wrong Python version, breaking the .pth bridge

**Why This Matters for Claude Code:**

When working on this template or projects created from it, remember that:
- VSCode's Ctrl+F5 runner uses `.venv/bin/python` to execute code
- If Python versions don't match, imports of container packages will fail
- The error message is misleading ("importing from source directory") but the root cause is binary incompatibility
- Always check Python versions first when debugging import errors

## known issues

See [doc/issues.md](doc/issues.md) for detailed issue tracking and improvements during active development.

## Research needed

1. Can we use HEIC image format with docling out of the box? If not is it relatively to work with HEIC images to get them ready for docling
2. I don't understand all the different options to use in docling and how they will affect output formatting and accuracy. There are several different OCR engines and well as models to use. Right now we are using the default but at some point I need to understand how to decide on a creating a good combination.
3. Which SmolLM3 model to use and what quantization level is acceptable

## Current Status & Next Steps

### ✅ Completed: Image Loading & Text Extraction (Phase 1)

**Package Structure:**
- `src/conference_reader/image_loader/` - Discovers JPG/JPEG images in directories
- `src/conference_reader/extraction/` - Extracts text and quality metrics using Docling
- `src/conference_reader/summarization/` - Stub for SmolLM3 integration (next phase)
- `src/conference_reader/output/` - Stub for CSV output (future phase)

**Key Decisions Made:**
1. **Export Format for LLM**: Use `export_to_markdown()` from Docling (not text/html/dict)
   - Preserves document structure (headings, lists)
   - LLMs are trained on markdown
   - Only ~3% larger than plain text
   - See: [doc/DOCLING_METRICS_GUIDE.md](doc/DOCLING_METRICS_GUIDE.md)

2. **Quality Metrics Tracking**: ProcessedDocument includes:
   - `quality_grade`: "EXCELLENT"/"GOOD"/"FAIR"/"POOR" (primary - stable API)
   - `quality_score`: 0.0-1.0 (for statistical analysis - unstable API)
   - `low_quality_grade`: Conservative quality estimate (worst areas)
   - `low_score`: 5th percentile score
   - `ocr_score`: OCR accuracy (0.0-1.0)
   - `layout_score`: Layout detection quality (0.0-1.0)
   - **Rationale**: Track both grades (stable, user-friendly) and scores (analysis/debugging)
   - **Usage**: Grades for filtering/production logic, scores for debugging/analysis

3. **Test Results**: All 6 test images processed successfully
   - 5 images: GOOD quality (0.800-0.899)
   - 1 image: EXCELLENT quality (0.911)

### 🔄 Next Phase: Text Summarization with SmolLM3

**Objective**: Implement TextSummarizer to generate concise summaries of extracted text

**Key Implementation Questions to Resolve:**

1. **Model Selection:**
   - Which SmolLM3 model? (135M, 360M, 1.7B, or 3B parameters)
   - What quantization level? (4-bit, 8-bit, or full precision)
   - Memory constraints: Framework 13 has 48GB RAM (generous headroom)
   - See: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

2. **Model Loading Strategy:**
   - Load model once in `__init__()` vs on-demand?
   - GPU vs CPU inference?
   - Use HuggingFace transformers pipeline vs direct model loading?

3. **Prompt Engineering:**
   - What prompt template works best for conference poster summaries?
   - Target summary length (1-2 sentences? 50 words? 100 words?)
   - Include extracted title in prompt context?
   - Example target output: "A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given."

4. **API Design (Already Decided):**
   - `summarize_single(doc: ProcessedDocument) -> ProcessedDocument`
   - `summarize_batch(docs: List[ProcessedDocument]) -> List[ProcessedDocument]`
   - Return new ProcessedDocument with `summary` field populated (immutable pattern)
   - **Need to add**: `summary: Optional[str] = None` field to ProcessedDocument

5. **Error Handling:**
   - What if summarization fails for one image?
   - Set `summary = None` and continue? Or flag as error?
   - Should we track summarization quality/confidence?

6. **Performance Considerations:**
   - Batch processing for efficiency?
   - Max token length handling (what if extracted text is huge?)
   - Should we truncate input text? If so, how? (First N chars? Smart truncation?)

**Files to Modify:**
1. `src/conference_reader/extraction/processed_document.py` - Add `summary` field
2. `src/conference_reader/summarization/text_summarizer.py` - Implement SmolLM3 integration
3. `main.py` - Add summarization step, display summaries in output
4. `pyproject.toml` - Add transformers/torch dependencies (if not already present)

**Research Needed Before Implementation:**
- Test SmolLM3 model loading and inference speed
- Benchmark memory usage with 3B model
- Experiment with prompt templates for optimal summary quality
- Determine optimal max input token length

**Reference Documentation:**
- SmolLM3 Model Card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- SmolLM3 Blog Post: https://huggingface.co/blog/smollm3
- Transformers Pipeline Docs: https://huggingface.co/docs/transformers/main_classes/pipelines

**Test Data Available:**
- 6 conference poster images in `/data/neurips/poster_test/`
- All have extracted markdown text (2,000-6,000 characters)
- Quality grades: 5 GOOD, 1 EXCELLENT

