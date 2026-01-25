# Qwen3-VL-4B GPU Hang Troubleshooting on AMD ROCm

**Date**: 2026-01-24 (initial), 2026-01-25 (solved)
**Hardware**: AMD Ryzen AI Max+ 395 with Radeon 8060S (gfx1151 / RDNA 3.5)
**ROCm Version Tested**: 7.1 (failed), 7.2 (✅ working with workarounds)
**Problem**: GPU hangs when running Qwen3-VL-4B with transformers library
**Status**: ✅ **SOLVED** - See "ROCm 7.2 Test Results" section below

---

## Problem Summary

### ROCm 7.1 Results (Failed)

| Mode | Result |
|------|--------|
| `auto` (SDPA) | GPU Hang |
| `eager_bfloat16` | GPU Hang |
| `eager_float16` | Initial success, then crashed on full dataset |
| `eager_float32` | GPU Hang |

### ROCm 7.2 Results (With Workarounds)

| Mode | Result |
|------|--------|
| `auto` (SDPA) | Not tested (SDPA disabled) |
| `eager_bfloat16` | GPU Hang (bf16 kernel bug) |
| `eager_float16` | ✅ **WORKS** - 100% accuracy, ~0.5s/image |
| `eager_float32` | OOM (attention matrix too large) |

**Error Message** (when crashes occur): `HW Exception by GPU node-1 (Agent handle: 0x...) reason: GPU Hang`

---

## Root Cause Analysis

Research reveals this is a **known issue** with Qwen vision models on AMD ROCm:

1. **Direct Issue Report**: [GPU Hang on RX 7800 XT when QWEN3-VL multimodal activated](https://github.com/vllm-project/vllm/issues/27499) (vLLM Issue #27499)

2. **Flash Attention Incompatibility**: [Qwen3 vs Qwen3-2507 Regression caused by flash attention on AMD ROCm](https://github.com/ollama/ollama/issues/12432) - Qwen2_5_VisionAttention uses `_Backend.TORCH_SDPA`, which RocmPlatform doesn't properly support

3. **ROCm Stability Issues**: Multiple reports of vision transformer models having stability issues on consumer AMD GPUs (RX 7000 series, Radeon AI series)

---

## GPU Options to Try (Before CPU Fallback)

### Option 1: PyTorch TunableOp ⭐ RECOMMENDED FIRST

**What it does**: Optimizes GEMM kernel selection specifically for your GPU hardware

**Performance Impact**: [AMD reports 22.9% speedup](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) and improved stability

**Implementation**:
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_VERBOSE=1

uv run python scripts/experiment_qwen_vl.py
```

**Expected Behavior**:
- First run: Slower (benchmarking all GEMM kernels)
- Creates: `tunableop_results.csv` with optimal kernel selections
- Subsequent runs: Use optimized kernels (faster + more stable)

**Why this might work**: Avoids problematic kernel code paths that cause GPU hangs

**References**:
- [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html)
- [TunableOp PyTorch Documentation](https://docs.pytorch.org/docs/stable/cuda.tunable.html)
- [ROCm TunableOp Fix PR](https://github.com/pytorch/pytorch/pull/142274)

---

### Option 2: Reduce Memory Pressure

**What it does**: Lowers GPU memory controller stress by reducing generation length

**Implementation**: Modify `scripts/experiment_qwen_vl.py`

```python
# In classify_image() function, change generate() call:
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,  # REDUCED from 512
    do_sample=False,
    num_beams=1,  # Disable beam search
    use_cache=True,  # Enable KV cache
)
```

**Why this might work**:
- Qwen3-VL issue [#180](https://github.com/QwenLM/Qwen3-VL/issues/180) shows memory reduction prevents OOM crashes
- Lower memory pressure = fewer memory controller conflicts
- Our classification task only needs 1-2 word responses ("poster" or "qr")

**Trade-off**: None for our use case - we don't need 512 tokens for binary classification

---

### Option 3: ROCm Stability Environment Variables

**What it does**: Uses more conservative/stable GPU runtime settings

**Implementation**:
```bash
export HSA_ENABLE_SDMA=0        # Use compute shaders instead of DMA engines
export HSA_ENABLE_INTERRUPT=0   # Use polling instead of interrupts
export ROCR_VISIBLE_DEVICES=0   # Explicit GPU device isolation
export VLLM_USE_TRITON_FLASH_ATTN=0  # Use CK flash attention (more stable on ROCm)
```

**Why this might work**:
- Per [ROCm environment variables docs](https://rocm.docs.amd.com/en/latest/reference/env-variables.html), these disable hardware features that can be unstable
- [HIP debugging docs](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html) recommend these for stability troubleshooting

**Trade-off**: Slightly slower performance, but more stable

---

### Option 4: Switch to Qwen2.5-VL-3B (Smaller Model)

**What it does**: Uses a 3B parameter model instead of 4B

**Implementation**: Change model in `scripts/experiment_qwen_vl.py`
```python
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # Changed from Qwen3-VL-4B-Instruct
```

**Why this might work**:
- Smaller model = less compute/memory stress
- May avoid triggering problematic code paths
- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) is officially supported

**Trade-off**: Potentially lower accuracy (but may be sufficient for poster vs QR classification)

---

### Option 5: AMD-Optimized Alternative - Instella-VL-1B

**What it does**: Use AMD's purpose-built VLM instead of Qwen

**Model**: [Instella-VL-1B](https://rocm.blogs.amd.com/artificial-intelligence/Instella-BL-1B-VLM/README.html) - 1.5B parameter VLM

**Why this is compelling**:
- **Trained specifically on AMD MI300X GPUs** with ROCm optimization
- Outperforms LLaVa-OneVision and MiniCPM-V2 on general benchmarks
- Much smaller (1.5B vs 4B) = better stability on consumer GPUs
- Native ROCm support (no CUDA→ROCm translation layer issues)

**Implementation**: Would require:
1. Testing model on poster classification task
2. Adapting prompt format
3. Modifying `experiment_qwen_vl.py` to load Instella model

**Trade-off**:
- Unknown accuracy on our specific task
- Would need prompt engineering
- Smaller model may miss subtle details

**Alternative Small VLMs with ROCm Support**:
- InternVL2-2B / InternVL2-4B
- H2OVL Mississippi-2B
- PaliGemma-3B
- SmolVLM2-2.2B

---

### Option 6: Upgrade to ROCm 7.2 ⭐ TRYING FIRST

**What it does**: Use latest ROCm container with potential stability fixes

**Container**: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.13_pytorch_*`

**Why this might work**:
- ROCm 7.1 → 7.2 likely includes bug fixes for vision models
- Newer PyTorch integration may have Qwen3-VL patches
- Flash attention backend improvements

**Implementation**: Update `.devcontainer/devcontainer.json`

**Expected**: If 7.2 doesn't fix it, proceed with Options 1-5 above

---

## Recommended Action Plan

### Phase 1: ROCm 7.2 Upgrade (Current)
1. Update devcontainer to ROCm 7.2
2. Test Qwen3-VL-4B with `eager_float16` mode
3. If still crashes → proceed to Phase 2

### Phase 2: Combined Optimization Approach
Try these **together** for maximum stability:

```bash
# Set environment variables
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_VERBOSE=1
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export ROCR_VISIBLE_DEVICES=0
export VLLM_USE_TRITON_FLASH_ATTN=0

# Run with eager_float32 + reduced tokens
uv run python scripts/experiment_qwen_vl.py
```

And modify `classify_image()` to use `max_new_tokens=128`

### Phase 3: Model Alternatives
1. Try Qwen2.5-VL-3B with all optimizations from Phase 2
2. Try Instella-VL-1B (AMD-native)
3. Try InternVL2-4B

### Phase 4: CPU Fallback
Only if all GPU options fail:
```python
DEVICE_MODE = "cpu"  # In scripts/experiment_qwen_vl.py
```

Expected performance: ~60-120 seconds per image (vs 13s on GPU)

---

## Technical Deep Dive

### Why SDPA Attention Backend Causes Issues

From research:
- Qwen2_5_VisionAttention hardcoded to use `_Backend.TORCH_SDPA`
- ROCm's SDPA implementation is **experimental** and unstable on consumer GPUs
- PyTorch's `eager` attention works but disables optimizations
- This is why `eager_float16` worked initially but crashed under load

### Memory Management Issues

Per [Qwen3-VL memory optimization discussion](https://github.com/QwenLM/Qwen3-VL/issues/180):
- Vision models calculate logits for **all token positions** during prefill
- This wastes memory significantly
- Chunked prefill can help: `--chunked-prefill-size 131072`
- But transformers library doesn't expose this easily (vLLM feature)

### ROCm vs CUDA Attention Backends

| Backend | CUDA | ROCm Status |
|---------|------|-------------|
| Flash Attention 2 | Stable | Experimental (Triton), unstable on RDNA3/Radeon AI |
| SDPA | Stable | Experimental, causes GPU hangs |
| Eager | Stable | Stable (but slow) |
| CK Flash Attention | N/A | More stable than Triton on ROCm |

**Solution**: Force `VLLM_USE_TRITON_FLASH_ATTN=0` to use CK implementation

---

## Relevant Hardware Context

**Your System**:
- CPU: AMD Ryzen AI Max+ 395
- GPU: Radeon 8060S (likely RDNA3 architecture)
- RAM: 48GB+
- ROCm: 7.1 → 7.2 (testing)

**Known Compatible AMD GPUs** (per documentation):
- AMD Instinct MI200 series (CDNA2) - best support
- AMD Instinct MI300X (CDNA3) - excellent support
- AMD Radeon RX 7000 series (RDNA3) - experimental support
- AMD Ryzen AI (RDNA3 iGPU) - experimental support ⚠️ **YOUR HARDWARE**

Your Radeon 8060S is in the **experimental support** category, which explains the stability issues.

---

## Research Sources

### Primary Issues:
- [GPU Hang on RX 7800 XT with QWEN3-VL (vLLM #27499)](https://github.com/vllm-project/vllm/issues/27499)
- [Qwen3 Flash Attention Regression on AMD ROCm (Ollama #12432)](https://github.com/ollama/ollama/issues/12432)
- [Qwen3-VL Memory Optimization (QwenLM #180)](https://github.com/QwenLM/Qwen3-VL/issues/180)

### ROCm Documentation:
- [PyTorch TunableOp for ROCm](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html)
- [ROCm Environment Variables](https://rocm.docs.amd.com/en/latest/reference/env-variables.html)
- [HIP Environment Variables](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)
- [vLLM V1 Performance Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)

### Alternative Models:
- [AMD Instella-VL-1B](https://rocm.blogs.amd.com/artificial-intelligence/Instella-BL-1B-VLM/README.html)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [OCR with Vision-Language Models on ROCm](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html)

### Performance & Compatibility:
- [vLLM on AMD ROCm GPUs](https://medium.com/@trademamba/serving-large-language-models-with-vllm-on-amd-rocm-gpus-a00ea352e2ac)
- [vLLM Installation with ROCm](https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html)
- [AMD ROCm First-Class Platform in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)

---

## Current Script Configuration

**File**: `scripts/experiment_qwen_vl.py`

```python
# Current settings (as of 2026-01-24)
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE_MODE = "eager_float32"  # Last tested mode before ROCm 7.2 upgrade
IMAGE_DIR = Path("/data/neurips/invalid_poster_images")
GROUND_TRUTH_FILE = Path("datasets/eval/validator_ground_truth.json")

# Classification prompt (working, do not change)
CLASSIFICATION_PROMPT = """Analyze this image and determine if it shows:
A) A FULL conference poster...
B) A CROPPED SECTION...
Your one-word answer:"""
```

**Test Dataset**: 11 images with ground truth labels (true=poster, false=qr crop)

---

## Next Steps After ROCm 7.2 Test

1. **If 7.2 works**: Document successful configuration, run full evaluation
2. **If 7.2 fails**: Implement Phase 2 (combined optimizations)
3. **If all GPU modes fail**: Switch to CPU mode and document performance
4. **Alternative path**: Test AMD Instella-VL-1B as replacement model

---

## ROCm 7.2 Test Results ✅ SOLVED

**Date**: 2026-01-25
**ROCm Version**: 7.2
**PyTorch Version**: 2.9.1
**Container**: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`

### Working Configuration

After extensive testing, the following configuration achieves **100% accuracy** with **~0.5 seconds per image**:

```python
# scripts/experiment_qwen_vl.py
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE_MODE = "eager_float16"  # NOT bfloat16 - bf16 hangs on Strix Halo

# Image resolution limits (critical for memory)
MIN_PIXELS = 256 * 28 * 28  # ~200K pixels
MAX_PIXELS = 512 * 28 * 28  # ~400K pixels
```

### Required Environment Variables

All set in `scripts/run_stable.sh`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_ENABLE_SDMA` | `0` | Use compute shaders instead of DMA engines |
| `HSA_ENABLE_INTERRUPT` | `0` | Use polling instead of interrupts |
| `ROCR_VISIBLE_DEVICES` | `0` | Explicit GPU isolation |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0` | Use CK flash attention |
| `TORCH_CUDNN_SDPA_ENABLED` | `0` | Disable unstable SDPA |
| `ROCBLAS_USE_HIPBLASLT` | `1` | Use optimized GEMM backend |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` | Use tuned kernels |

### Key Findings

#### BFloat16 is BROKEN on Strix Halo ⚠️

Despite ROCm 7.2 research recommending BF16 for optimal performance, **bfloat16 causes GPU hangs** on Strix Halo (gfx1151):

| Precision | Result |
|-----------|--------|
| `bfloat16` | GPU hang in convolution kernel `naive_conv_ab_nonpacked_fwd_ncdhw_ushort` |
| `float16` | ✅ **WORKS** - stable, fast (~0.5s/image) |
| `float32` | OOM - attention matrix too large (needs 133 GiB) |

The crash occurs in the vision encoder regardless of image size, indicating a fundamental kernel bug in bf16 convolution on RDNA 3.5.

#### Image Resolution is Critical

High-resolution poster images create too many vision tokens:

| Max Pixels | Tokens | Attention Memory | Result |
|------------|--------|------------------|--------|
| Default (~1.6M) | ~12,000 | ~67 GiB | OOM |
| 512×28×28 (~400K) | ~3,000 | ~4 GiB | ✅ Works |

#### TunableOp Kernel Selection

PyTorch TunableOp found stable kernels stored in `tunableop_results0.csv`:
- 15 operations use `Gemm_Hipblaslt` (AMD's optimized library)
- 10 operations use `Gemm_Rocblas` (older but stable)
- 1 operation uses `Default` (PyTorch fallback)

### How to Run

```bash
# First time (with tuning):
./scripts/run_stable.sh --tuning

# Subsequent runs (uses cached kernels):
./scripts/run_stable.sh
```

### Performance Results

| Metric | Value |
|--------|-------|
| Precision | float16 |
| Inference time | ~0.5 seconds/image |
| Accuracy | 100% (11/11 test images) |
| GPU Memory | ~18 GiB peak |
| Model load time | ~5 seconds |

### Debugging Notes

When GPU hangs occurred, the stability environment variables prevented hard system crashes:
- Process received `SIGABRT` instead of hanging the system
- Core dumps showed crash in `libhsa-runtime64.so` (HSA runtime timeout)
- GPU driver recovered without requiring reboot (in most cases)

To debug future issues:
```bash
# On host machine (not in container):
sudo dmesg | grep -i -E "(amdgpu|gpu|hang|reset|error)"
coredumpctl info -1
```

---

**Document Version**: 2.0
**Last Updated**: 2026-01-25
**Status**: ✅ SOLVED - float16 + reduced resolution + stability env vars
