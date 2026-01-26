# Fixing OCR Failures in Docling: From RapidOCR to Tesseract

## The Problem

When processing conference poster images with Docling and RapidOCR, we encountered a frustrating issue: some images would fail silently, returning zero characters despite containing clear, readable text.

From processing 150 poster images:
- Images 1-6: Processed normally (3-12 seconds each)
- Image 7 (IMG_1171.JPEG): Empty OCR result, took 46 seconds
- Image 8 onwards: Mixed results with some timeouts

The key warning message from RapidOCR:
```
[WARNING] RapidOCR main.py:125: The text detection result is empty
```

## Investigation

### Step 1: Identify the Failing Images

We created a diagnostic script to test each image individually. The results showed two images consistently failing:

| Image | RapidOCR Result | Processing Time |
|-------|-----------------|-----------------|
| IMG_1171.JPEG | 0 chars | 46.34s |
| IMG_1209.JPEG | 0 chars | 46.40s |

### Step 2: Examine the Images

Visual inspection revealed both images were valid conference posters with plenty of readable text:

- **IMG_1171.JPEG**: "GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset"
- **IMG_1209.JPEG**: "TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields"

Both images had prominent QR codes, which we initially suspected might be confusing the text detection model.

### Step 3: Test Alternative OCR Engine

We tested Tesseract OCR directly on the failing images:

```bash
tesseract /data/neurips/posters/IMG_1171.JPEG stdout
```

Result: **2,372 characters extracted successfully**

| Image | RapidOCR | Tesseract |
|-------|----------|-----------|
| IMG_1171.JPEG | 0 chars | 2,372 chars |
| IMG_1209.JPEG | 0 chars | 1,819 chars |

## The Solution

### Switching to Tesseract in Docling

Docling supports multiple OCR backends. We switched from RapidOCR to Tesseract CLI:

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)

ocr_options = TesseractCliOcrOptions(lang=["eng"])

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=ocr_options,
)
```

### GPU Acceleration for Layout Detection

While Tesseract OCR is CPU-only, Docling's layout detection can still use GPU acceleration:

```python
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)

accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.AUTO,  # Detects CUDA or ROCm
    num_threads=4,
)

pipeline_options = PdfPipelineOptions(
    accelerator_options=accelerator_options,
    # ... other options
)
```

### ROCm Stability Settings for AMD GPUs

For AMD GPUs using ROCm, we apply stability settings before GPU operations:

```python
def apply_rocm_stability_settings():
    settings = {
        "HSA_ENABLE_SDMA": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "VLLM_USE_TRITON_FLASH_ATTN": "0",
        "TORCH_CUDNN_SDPA_ENABLED": "0",
    }
    for key, value in settings.items():
        os.environ[key] = value
```

## Results

After switching to Tesseract:

| Metric | Before (RapidOCR) | After (Tesseract) |
|--------|-------------------|-------------------|
| IMG_1171.JPEG | 0 chars | 33,776 chars |
| IMG_1209.JPEG | 0 chars | 1,684 chars |
| Full test set (12 images) | 2 failures | 0 failures |
| Success rate | ~83% | **100%** |

## Architecture Summary

| Component | Device | Notes |
|-----------|--------|-------|
| Layout detection | GPU (AUTO) | Uses CUDA or ROCm via HIP |
| Table detection | GPU (AUTO) | Uses CUDA or ROCm via HIP |
| OCR | CPU | Tesseract (no GPU support) |

## Key Takeaways

1. **RapidOCR's text detection can fail silently** on certain images, even when they contain clear text.

2. **Tesseract is more robust** for diverse image types, though it lacks GPU acceleration.

3. **The suspiciously consistent timing** (both failures at ~46 seconds) suggested they were hitting the same failure path in RapidOCR's text detection stage.

4. **GPU acceleration is still valuable** for layout detection, even when OCR runs on CPU.

5. **Always have a fallback strategy** - testing with alternative OCR engines revealed the issue was with RapidOCR, not the images themselves.

---

## Part 2: Pre-filtering Images with a Vision Language Model

### The QR Code Problem

Not all photos in our conference image collection were full posters. Some were cropped close-ups showing only QR codes—taken so attendees could quickly scan them later. When these images hit the OCR pipeline:

- Docling would successfully extract... a QR code's gibberish
- Processing time was wasted on non-poster content
- The summarization step would produce meaningless output

**Example**: IMG_1223.JPEG was a zoomed-in photo of just QR codes, not a complete poster.

### Why Docling Couldn't Filter These

We initially hoped Docling's quality metrics could distinguish posters from QR code crops:

| Metric | Full Poster | QR Code Crop |
|--------|-------------|--------------|
| text_length | 1000-30000 chars | 10-500 chars |
| heading_count | 2-10 | 0-1 |
| children_count | 10-100 | 1-5 |

The problem: **significant overlap**. Some valid posters with image-heavy layouts had low text extraction (e.g., IMG_1222 extracted only 30 characters despite being a valid poster). Heuristic thresholds produced 36-73% accuracy depending on configuration—unacceptable for production.

### The VLM Solution

We needed semantic understanding: "Is this a complete conference poster, or just a cropped section showing QR codes?"

This is a perfect task for a Vision Language Model (VLM). We chose **Qwen3-VL-4B** for its balance of accuracy and efficiency:

```python
CLASSIFICATION_PROMPT = """Analyze this image and determine if it shows:
A) A FULL conference poster - complete with title, sections, structure
B) A CROPPED SECTION - mainly QR codes, missing main content

Respond with ONLY ONE WORD: "poster" or "qr"
"""
```

The VLM achieved **100% accuracy** on our test set of 11 labeled images.

---

## Part 3: The ROCm Nightmare (and How We Solved It)

### The Hardware

- **GPU**: AMD Radeon 8060S (RDNA 3.5 / gfx1151)
- **APU**: AMD Ryzen AI Max+ 395 (Strix Halo)
- **Software**: ROCm 7.1 → 7.2, PyTorch 2.9.1

This is cutting-edge AMD hardware with "experimental" ROCm support. What followed was a multi-day battle with GPU hangs.

### Initial Attempts with ROCm 7.1

Every configuration we tried resulted in GPU hangs:

| Mode | Result |
|------|--------|
| `auto` (SDPA attention) | GPU Hang |
| `eager_bfloat16` | GPU Hang |
| `eager_float16` | Initial success, then crashed on full dataset |
| `eager_float32` | GPU Hang |

The error message was always the same:
```
HW Exception by GPU node-1 (Agent handle: 0x...) reason: GPU Hang
```

### Root Cause: SDPA Attention is Broken on Consumer AMD GPUs

Research revealed the issue: Qwen's vision attention uses PyTorch's Scaled Dot Product Attention (SDPA), which is **experimental and unstable** on ROCm.

From [vLLM Issue #27499](https://github.com/vllm-project/vllm/issues/27499):
> "GPU Hang on RX 7800 XT when QWEN3-VL multimodal activated"

From [Ollama Issue #12432](https://github.com/ollama/ollama/issues/12432):
> "Qwen3 vs Qwen3-2507 Regression caused by flash attention on AMD ROCm"

### The Solution: ROCm 7.2 + Stability Environment Variables + float16

After upgrading to ROCm 7.2 and extensive testing, we found a working configuration:

#### 1. Critical Environment Variables

```bash
export HSA_ENABLE_SDMA=0           # Use compute shaders instead of DMA
export HSA_ENABLE_INTERRUPT=0      # Use polling instead of interrupts
export ROCR_VISIBLE_DEVICES=0      # Explicit GPU isolation
export VLLM_USE_TRITON_FLASH_ATTN=0  # Use CK flash attention
export TORCH_CUDNN_SDPA_ENABLED=0  # Disable unstable SDPA
export ROCBLAS_USE_HIPBLASLT=1     # Use optimized GEMM backend
export PYTORCH_TUNABLEOP_ENABLED=1 # Use tuned kernels
```

#### 2. Float16, NOT BFloat16

Despite AMD documentation recommending bfloat16, it's **broken on Strix Halo**:

| Precision | Result |
|-----------|--------|
| `bfloat16` | GPU hang in convolution kernel |
| `float16` | ✅ **WORKS** - stable, ~0.5s/image |
| `float32` | OOM - needs 133 GiB for attention matrix |

The crash occurred in `naive_conv_ab_nonpacked_fwd_ncdhw_ushort`—a fundamental bf16 convolution kernel bug on RDNA 3.5.

#### 3. Image Resolution Limits

High-resolution poster images create too many vision tokens:

| Max Pixels | Vision Tokens | Attention Memory | Result |
|------------|---------------|------------------|--------|
| Default (~1.6M) | ~12,000 | ~67 GiB | OOM |
| 512×28×28 (~400K) | ~3,000 | ~4 GiB | ✅ Works |

```python
processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=256 * 28 * 28,  # ~200K pixels
    max_pixels=512 * 28 * 28,  # ~400K pixels
)
```

#### 4. PyTorch TunableOp for Kernel Selection

TunableOp benchmarks GEMM kernels and selects stable ones for your hardware:

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1  # First run only
```

This creates `tunableop_results.csv` with optimal kernel selections, avoiding problematic code paths.

### Final Working Configuration

```python
class VLMBackend:
    def __init__(self, device_mode="eager_float16"):
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "attn_implementation": "eager",  # NOT SDPA
        }

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        )
```

### Performance Results

| Metric | Value |
|--------|-------|
| Precision | float16 |
| Inference time | ~0.5 seconds/image |
| Accuracy | 100% (11/11 test images) |
| GPU Memory | ~18 GiB peak |

---

## Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Image Processing Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Raw Images │───▶│ VLM Classifier│───▶│    Posters   │  │
│  │  (150 photos)│    │ (Qwen3-VL-4B)│    │   Only (~80) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                     │         │
│                             │                     ▼         │
│                     Filters out          ┌──────────────┐  │
│                     QR code crops        │   Docling    │  │
│                                          │  (Tesseract) │  │
│                                          └──────────────┘  │
│                                                   │         │
│                                                   ▼         │
│                                          ┌──────────────┐  │
│                                          │  Extracted   │  │
│                                          │    Text      │  │
│                                          └──────────────┘  │
└─────────────────────────────────────────────────────────────┘

Hardware Utilization:
- VLM Classification: GPU (ROCm, float16, eager attention)
- Layout Detection: GPU (ROCm via AcceleratorDevice.AUTO)
- OCR: CPU (Tesseract CLI)
```

---

## Lessons Learned

### 1. Don't Trust "Experimental" GPU Support
AMD ROCm on consumer GPUs (RX 7000, Radeon AI) is marked experimental for a reason. Expect to spend significant time on stability issues.

### 2. Environment Variables Are Critical
On ROCm, the difference between "works perfectly" and "GPU hang" can be a single environment variable (`HSA_ENABLE_SDMA=0`).

### 3. Precision Matters More Than You Think
bfloat16 vs float16 isn't just about accuracy—on some hardware, one works and the other crashes.

### 4. VLMs Are Great for Semantic Filtering
When heuristics fail (36-73% accuracy), a VLM can achieve 100% accuracy on classification tasks with minimal latency (~0.5s/image).

### 5. Fallback OCR Engines Save Projects
RapidOCR was faster but unreliable. Tesseract is slower but robust. Having alternatives is essential.

---

## References

- [Docling Pipeline Options](https://docling-project.github.io/docling/reference/pipeline_options/)
- [Docling GPU Support](https://docling-project.github.io/docling/usage/gpu/)
- [Tesseract OCR with Docling](https://docling-project.github.io/docling/examples/tesseract_lang_detection/)
- [GPU Hang on RX 7800 XT with QWEN3-VL (vLLM #27499)](https://github.com/vllm-project/vllm/issues/27499)
- [Qwen3 Flash Attention Regression on AMD ROCm (Ollama #12432)](https://github.com/ollama/ollama/issues/12432)
- [PyTorch TunableOp for ROCm](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html)
- [ROCm Environment Variables](https://rocm.docs.amd.com/en/latest/reference/env-variables.html)
