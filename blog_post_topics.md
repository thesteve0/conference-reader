# Blog Post Topics from Conference Poster Summarization Project

This document captures potential blog post topics and lessons learned from building a VLM-based pipeline for extracting and summarizing conference poster images.

## Technical Deep Dives

### 1. Running VLMs on AMD GPUs: A ROCm Survival Guide

- The challenges of getting Qwen3-VL and other models working on ROCm/Strix Halo
- Environment variables that matter: `HSA_ENABLE_SDMA=0`, `TORCH_CUDNN_SDPA_ENABLED=0`, etc.
- Why you need `attn_implementation="eager"` and can't use flash attention (yet)
- The tunable ops caching system for performance
- Disabling cuDNN/MIOpen for stability

### 2. A Practical Pipeline for Conference Poster Summarization with Local VLMs

- The full architecture: classify -> OCR -> summarize
- Why you need a classification step (QR codes vs posters)
- Choosing the right model for each stage (Qwen3-VL for vision, SmolLM3 for text)
- Memory management when chaining multiple models
- Exporting structured results to CSV

### 3. OCR Engine Shootout: EasyOCR vs RapidOCR vs Tesseract for Conference Posters

- We tried multiple OCR approaches during development
- RapidOCR failures on certain images
- Why EasyOCR won out for this use case
- The tradeoffs between pure VLM extraction vs dedicated OCR
- Image scaling strategies for better OCR accuracy

## Lessons Learned

### 4. Why VLM-Only Poster Reading Doesn't Work (Yet)

- The initial dream: feed poster image directly to VLM, get summary
- Reality: VLMs hallucinate titles and produce generic summaries
- The hybrid approach: OCR for text extraction + LLM for summarization
- When VLMs shine (classification) vs where they struggle (dense OCR)
- Token limits when processing large amounts of extracted text

### 5. Managing GPU Memory When Chaining AI Models

- Loading/unloading models sequentially (`classifier.unload()`)
- The `_reset_reader()` pattern after slow extractions (>60s threshold)
- `torch.cuda.empty_cache()` and garbage collection between models
- Why you can't just load 3 models simultaneously on consumer hardware
- Detecting and recovering from GPU resource exhaustion

### 6. Prompt Engineering for Poster Summarization: Fighting Example Parroting

- Why few-shot prompting can backfire with smaller models
- Explicitly telling the model "DO NOT repeat this example"
- Token limits and truncation strategies (`MAX_INPUT_CHARS = 4000`)
- Balancing prompt length with output quality
- Deterministic generation (`do_sample=False`) for reproducibility

## Feasibility Assessment

### The Honest Take on VLM-Based Poster Summarization

#### What Worked Well

- **Image classification (poster vs QR)** - Near 100% accuracy with simple prompting
- **The pipeline architecture** - Clean separation of concerns between classification, extraction, and summarization
- **Local inference on consumer AMD GPU** - No API costs, full privacy
- **EasyOCR text extraction** - Reliable and fast enough for batch processing

#### What Was Harder Than Expected

- **VLMs struggle with dense, structured text extraction** - Conference posters have complex layouts that confuse pure vision models
- **ROCm compatibility required significant tuning** - Many environment variables and attention implementation changes needed
- **Memory management across multiple models is non-trivial** - Had to implement explicit unload/reload patterns
- **Small VLMs produce generic summaries** - Without strong prompting constraints, outputs were too vague
- **Some images cause OCR hangs** - Needed timeout handling and reader reset logic

#### Verdict

**Feasible but hybrid approach wins.** Use VLMs for what they're good at (understanding visual layout, classification) and dedicated OCR + text LLMs for extraction and summarization. The three-stage pipeline (classify -> extract -> summarize) proved more reliable than attempting end-to-end VLM processing.

## Project Architecture Reference

```
conference-reader/
├── main.py                           # Full pipeline orchestration
├── src/conference_reader/
│   ├── classifier/
│   │   ├── image_classifier.py       # Qwen3-VL poster vs QR classification
│   │   └── vlm_backend.py            # ROCm-optimized VLM wrapper
│   ├── extraction/
│   │   └── document_extractor.py     # EasyOCR text extraction
│   ├── summarization/
│   │   └── text_summarizer.py        # SmolLM3 text summarization
│   └── config/
│       └── rocm_config.py            # AMD GPU stability settings
```

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| Classification | Qwen3-VL-4B-Instruct | Distinguish posters from QR code crops |
| Text Extraction | EasyOCR | Extract text from poster images |
| Summarization | SmolLM3-3B | Generate 1-2 sentence summaries |

## Key Configuration Insights

### ROCm Stability Settings

```python
settings = {
    "HSA_ENABLE_SDMA": "0",
    "HSA_ENABLE_INTERRUPT": "0",
    "ROCR_VISIBLE_DEVICES": "0",
    "VLLM_USE_TRITON_FLASH_ATTN": "0",
    "TORCH_CUDNN_SDPA_ENABLED": "0",
    "ROCBLAS_USE_HIPBLASLT": "1",
    "PYTORCH_TUNABLEOP_ENABLED": "1",
}
```

### VLM Loading for ROCm

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",  # Required for ROCm stability
)
```
