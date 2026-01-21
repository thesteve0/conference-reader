# SmolLM3-3B Experimentation Results

**Date:** 2026-01-14
**Model:** HuggingFaceTB/SmolLM3-3B
**Task:** Conference poster summarization

## Executive Summary

Successfully tested SmolLM3-3B for generating 1-2 sentence summaries of conference posters. The model produces coherent, informative summaries with acceptable performance on AMD GPU (ROCm).

**Recommendation:** Proceed with SmolLM3-3B using Template V2 (example-driven) approach, but need to address the example-copying issue.

## Hardware & Performance

### System Configuration
- **GPU:** AMD Radeon 8060S
- **ROCm Version:** 7.1
- **PyTorch Version:** 2.9.1
- **Total RAM:** 48 GB

### Model Metrics
- **Parameters:** 3.08B (3,075,098,624)
- **GPU Memory:** 5.73 GB allocated/reserved
- **Load Time:** 4.76 seconds (cached), ~98 seconds (first load)
- **Inference Speed:** 10-12 tokens/second
- **Average Summary Time:** 6.70 seconds per document

### Performance Comparison: 135M vs 3B

| Metric | SmolLM2-135M | SmolLM3-3B |
|--------|--------------|------------|
| Parameters | 134.5M | 3,075M |
| GPU Memory | 0.25 GB | 5.73 GB |
| Load Time | 8.55s | 4.76s (cached) |
| Inference Speed | ~34 tok/s | ~11 tok/s |
| Summary Quality | Poor (repetitive) | Good (coherent) |

**Conclusion:** The 3B model is significantly better for summarization despite slower inference.

## Prompt Template Evaluation

### Template V1: Simple Instruction
```
Summarize the following conference poster in 1-2 concise sentences.

Title: {title}

Content:
{text}

Summary:
```

**Results:**
- ❌ Produces 3-4 sentence summaries (too verbose)
- ✅ Coherent and informative content
- ❌ Exceeds target length even with 80 token limit

**Example Output:**
> "The TLIGHT project aims to develop a multimodal remote sensing system for monitoring natural disasters, addressing the challenge of rapid event detection and progression analysis. The dataset MONITRS-QA is created by extracting event sequences from news articles, geocoding locations, and downloading satellite images. The system achieves high accuracy in temporal grounding and event classification, providing a useful benchmark for future research. The project is a"

**Assessment:** Too verbose for scanning purposes.

---

### Template V2: Example-Driven ⭐ BEST (with caveats)
```
You are helping someone quickly scan conference posters. Summarize the poster below in 1-2 sentences, focusing on the main research contribution and results.

Example:
Title: "CrypticBio: A large multimodal dataset"
Summary: A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given.

Title: {title}

Content:
{text}

Summary:
```

**Results:**
- ✅ Produces concise 2-3 sentence summaries
- ✅ Focuses on main contribution and results
- ❌ **CRITICAL ISSUE:** Sometimes just copies the example verbatim

**Good Example Outputs:**

1. **COLORBENCH poster:**
   > "COLORBENCH is a comprehensive benchmark for evaluating the performance of Visual Language Models (VLMs) in color perception, reasoning, and robustness. It consists of 1,448 instances and 5,814 image-text questions across 11 diverse tasks, addressing key questions about VLMs' ability to perceive and understand color."

2. **Tactile Imaging poster:**
   > "This poster presents a dataset of 1400 configurations, including MRI ground truth and 250 hours of tactile data, focusing on tactile representation for soft objects. It also introduces a conditional tactile imaging code and explores its applications in tactile imaging."

3. **AI Health Monitoring poster:**
   > "The poster emphasizes the importance of statistically valid post-deployment monitoring for AI-based digital health tools, highlighting the challenges of monitoring model performance due to limited access to labels and changing data distributions. It proposes a solution based on hypothesis testing to address these issues."

**Bad Example (copied the prompt example):**
> "A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given."

**Assessment:** Best template, but needs refinement to prevent example copying.

---

### Template V3: No Title
```
Summarize this conference poster in 1-2 sentences:

{text}

Summary:
```

**Results:**
- ❌ Produces 3-4 sentence summaries
- ✅ Coherent content
- ❌ Similar verbosity issues to V1

**Example Output:**
> "The TLIGHT project aims to improve disaster monitoring by leveraging multimodal remote sensing data and news articles. It introduces a new dataset, MONITRS, which combines satellite images and event descriptions from news articles. The dataset is created through a two-step process: first, extracting event sequences from news articles, and second, geocoding these locations to obtain precise event locations. The project's contributions include a"

**Assessment:** Too verbose, no advantage over V1.

## Key Findings

### What Works ✅
1. **Model Selection:** SmolLM3-3B produces coherent, informative summaries
2. **GPU Acceleration:** ROCm works perfectly with AMD Radeon 8060S
3. **Memory Footprint:** 5.73 GB is well within our 48 GB budget
4. **Inference Speed:** 6-7 seconds per summary is acceptable for production
5. **Content Quality:** Summaries capture main research contributions and results

### Issues to Address ❌
1. **Example Copying:** Template V2 sometimes copies the example instead of generating new content
2. **Length Control:** Even with 80 token limit, summaries can be 3-4 sentences
3. **Token Limit Truncation:** Some summaries get cut off mid-sentence at 80 tokens

### Observations
- The model cached after first load (98s → 4.8s on subsequent runs)
- Inference speed is consistent (~10-11 tok/s) across all documents
- Quality varies by poster complexity and OCR accuracy

## Recommendations for Implementation

### 1. Prompt Engineering
**Action:** Revise Template V2 to prevent example copying
- Option A: Remove the example entirely and rely on instruction
- Option B: Use multiple diverse examples to prevent pattern matching
- Option C: Add explicit instruction "DO NOT copy the example"

### 2. Token Length Tuning
**Action:** Experiment with different max_new_tokens values
- Try 60 tokens (force more concise summaries)
- Try 100 tokens (allow complete sentences, post-process to extract first 2)
- Implement sentence boundary detection and truncate after 2nd sentence

### 3. Post-Processing
**Action:** Add summary validation and cleanup
- Detect and reject summaries that match the example
- Truncate at sentence boundaries instead of token limits
- Validate summary length (target: 1-2 sentences, 20-50 words)

### 4. Batch Processing Optimization
**Action:** Consider batching for efficiency
- Current: Process documents sequentially (~7s each)
- Potential: Batch multiple documents in single forward pass
- Trade-off: Memory usage vs speed

## Next Steps

1. **Refine Prompt Template V2** to eliminate example copying
2. **Experiment with token limits** (60, 80, 100) to find optimal length
3. **Implement post-processing** to ensure quality and length constraints
4. **Add `summary` field** to ProcessedDocument dataclass
5. **Implement TextSummarizer class** using these findings
6. **Create unit tests** for summarization quality

## Test Data Summary

**6 conference posters processed:**
1. TLIGHT - Natural disaster monitoring dataset
2. COLORBENCH - VLM color perception benchmark
3. Tactile Imaging - MRI + tactile data for soft objects
4. AI Health Monitoring - Post-deployment monitoring for digital health
5. CrypticBio - Multimodal dataset for confusing species
6. CURE-Bench - Clinical reasoning multi-agent pipeline

All summaries were coherent and captured the main research contribution.

## Appendix: Full Configuration

```python
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
MAX_NEW_TOKENS = 80
DEVICE = "cuda"  # AMD GPU via ROCm
TORCH_DTYPE = torch.float16
MAX_INPUT_CHARS = 2000  # Truncate long posters
TEMPERATURE = None  # Deterministic (do_sample=False)
```

## References

- Model: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- SmolLM Blog: https://huggingface.co/blog/smollm3
- Test Data: `/data/neurips/poster_test/` (6 conference posters)
