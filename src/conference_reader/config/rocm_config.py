"""ROCm stability configuration for AMD GPU execution.

This module provides environment variable settings that ensure stable
execution of VLM models on ROCm/AMD GPUs, particularly for Strix Halo
chipsets (gfx1151).

Usage:
    from conference_reader.config import apply_rocm_stability_settings

    # Call before any GPU operations
    apply_rocm_stability_settings()
"""

import os


def apply_rocm_stability_settings() -> None:
    """Apply environment variables for stable ROCm execution.

    These settings are optimized for:
    - AMD Strix Halo chipsets (RYZEN AI MAX+ 395, RYZEN AI 9 HX Pro 370)
    - ROCm 7.1+ with PyTorch 2.9.1+
    - Qwen3-VL and similar vision-language models

    Settings applied:
    - HSA_ENABLE_SDMA=0: Disable SDMA for stability
    - HSA_ENABLE_INTERRUPT=0: Disable interrupts for stability
    - ROCR_VISIBLE_DEVICES=0: Use first GPU only
    - VLLM_USE_TRITON_FLASH_ATTN=0: Disable Triton flash attention
    - TORCH_CUDNN_SDPA_ENABLED=0: Disable SDPA (use eager attention)
    - ROCBLAS_USE_HIPBLASLT=1: Enable hipBLASLt for better performance
    - PYTORCH_TUNABLEOP_ENABLED=1: Enable tunable ops
    - PYTORCH_TUNABLEOP_FILENAME: Cache tuned kernels
    """
    # Fix deprecated PYTORCH_CUDA_ALLOC_CONF if present in environment
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        val = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_ALLOC_CONF"] = val

    settings = {
        "HSA_ENABLE_SDMA": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "VLLM_USE_TRITON_FLASH_ATTN": "0",
        "TORCH_CUDNN_SDPA_ENABLED": "0",
        "ROCBLAS_USE_HIPBLASLT": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_FILENAME": "tunableop_results.csv",
    }

    for key, value in settings.items():
        os.environ[key] = value