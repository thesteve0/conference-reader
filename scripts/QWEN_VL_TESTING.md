# Qwen3-VL-4B Testing Guide - ROCm GPU Workarounds

This guide explains how to test different configurations to work around AMD ROCm GPU compatibility issues.

## Quick Start

### Automated Testing (Recommended)

Run all modes automatically until one works:
```bash
./scripts/test_all_modes.sh
```

This will test in order:
1. ✨ **Option 4**: Experimental ROCm (fastest if it works)
2. 🔧 **Option 2**: Eager attention + float16
3. 🔧 **Option 3**: Eager attention + float32
4. 💻 **Option 1**: CPU only (slowest, most reliable)

### Manual Testing

If you prefer manual control, test each mode individually:

#### Try Option 4: Experimental ROCm (Fastest)

1. Edit `scripts/experiment_qwen_vl.py`:
   ```python
   DEVICE_MODE = "auto"
   ```

2. Run with experimental ROCm features:
   ```bash
   export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
   source .venv/bin/activate
   python scripts/experiment_qwen_vl.py
   ```

**If it hangs or crashes**, try Option 2.

---

#### Try Option 2: Eager Attention + Float16

1. Edit `scripts/experiment_qwen_vl.py`:
   ```python
   DEVICE_MODE = "eager_float16"
   ```

2. Run normally:
   ```bash
   source .venv/bin/activate
   python scripts/experiment_qwen_vl.py
   ```

**If it hangs or crashes**, try Option 3.

---

#### Try Option 3: Eager Attention + Float32

1. Edit `scripts/experiment_qwen_vl.py`:
   ```python
   DEVICE_MODE = "eager_float32"
   ```

2. Run normally:
   ```bash
   source .venv/bin/activate
   python scripts/experiment_qwen_vl.py
   ```

**If it hangs or crashes**, use Option 1.

---

#### Option 1: CPU Only (Most Reliable)

1. Edit `scripts/experiment_qwen_vl.py`:
   ```python
   DEVICE_MODE = "cpu"
   ```

2. Run normally:
   ```bash
   source .venv/bin/activate
   python scripts/experiment_qwen_vl.py
   ```

This should **always work**, but will be slow (~80-120 seconds per image).

---

## Mode Comparison

| Mode | Speed | Memory | Reliability | Notes |
|------|-------|--------|-------------|-------|
| Option 4: auto | ⚡⚡⚡ Fastest | ~8GB | ⚠️ Experimental | May hang on AMD GPUs |
| Option 2: eager_float16 | ⚡⚡ Fast | ~8GB | ⚠️ Uncertain | Disables optimized attention |
| Option 3: eager_float32 | ⚡ Medium | ~16GB | ⚠️ Uncertain | High memory usage |
| Option 1: cpu | 🐌 Slow | ~8GB RAM | ✅ Reliable | Works everywhere |

## Understanding the Issue

**Problem**: AMD Radeon 8060S with ROCm 7.1 has experimental/unstable support for Scaled Dot Product Attention (SDPA) used by Qwen3-VL.

**Symptoms**:
- GPU hang during inference
- Error: `HW Exception by GPU node-1 reason: GPU Hang`
- Warnings about "Flash Efficient attention" being experimental

**Root Causes**:
1. SDPA attention implementation not fully stable on ROCm
2. Auto dtype selection may choose incompatible precision
3. Vision-language models less tested on AMD GPUs vs NVIDIA

## Expected Performance

### GPU (if working):
- Model load: ~15-30 seconds
- Inference: ~5-20 seconds per image
- Total for 2 images: ~30-60 seconds

### CPU:
- Model load: ~5-10 seconds
- Inference: ~80-120 seconds per image
- Total for 2 images: ~160-240 seconds

## Troubleshooting

### GPU still hanging?
- Make sure you're using the correct DEVICE_MODE
- Try clearing GPU memory: `sudo rocm-smi --resetfans && sudo rocm-smi --resetclocks`
- Check GPU isn't being used by other processes: `rocm-smi`

### Out of memory?
- Try `DEVICE_MODE = "cpu"`
- Close other applications using GPU
- Option 3 (float32) uses ~2x memory of float16

### Import errors?
- Make sure venv is activated: `source .venv/bin/activate`
- Check dependencies: `uv pip list | grep -E "transformers|torch|pillow"`

## What to Report

If you find a working configuration, please note:
- Which DEVICE_MODE worked
- GPU model (AMD Radeon 8060S)
- ROCm version: 7.1
- PyTorch version: 2.9.1+rocm7.1.0
- Inference time per image

This helps improve compatibility for others!
