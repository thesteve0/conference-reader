#!/bin/bash
# ROCm Stability Wrapper for Qwen-VL experiments
# Based on QWEN_VL_ROCM_STABILITY_RESEARCH.md findings
#
# Usage: ./scripts/run_stable.sh [--tuning] [--debug]
#
# Options:
#   --tuning   Enable PyTorch TunableOp kernel tuning (slower first run, faster after)
#   --debug    Enable verbose debugging output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
ENABLE_TUNING=0
ENABLE_DEBUG=0
for arg in "$@"; do
    case $arg in
        --tuning) ENABLE_TUNING=1 ;;
        --debug) ENABLE_DEBUG=1 ;;
    esac
done

echo "=============================================="
echo "ROCm Stability Wrapper for Qwen-VL"
echo "=============================================="
echo ""

# ROCm 7.2 Performance Settings (already in devcontainer.json, but ensure they're set)
export ROCBLAS_USE_HIPBLASLT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ROCm Stability Settings - Use conservative GPU features
# These disable hardware features that can be unstable on consumer GPUs
export HSA_ENABLE_SDMA=0          # Use compute shaders instead of DMA engines
export HSA_ENABLE_INTERRUPT=0     # Use polling instead of interrupts (more stable)
export ROCR_VISIBLE_DEVICES=0     # Explicit GPU device isolation

# Flash Attention - Use CK implementation instead of Triton (more stable on ROCm)
export VLLM_USE_TRITON_FLASH_ATTN=0

# Disable SDPA which is known to cause GPU hangs with Qwen VL models
export TORCH_CUDNN_SDPA_ENABLED=0

# PyTorch TunableOp - Optimizes GEMM kernel selection for your specific GPU
if [ "$ENABLE_TUNING" -eq 1 ]; then
    echo "[TUNING] PyTorch TunableOp ENABLED"
    echo "         First run will be slower while benchmarking kernels"
    echo "         Results cached in: tunableop_results.csv"
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_TUNING=1
    export PYTORCH_TUNABLEOP_VERBOSE=1
    export PYTORCH_TUNABLEOP_FILENAME="${PROJECT_DIR}/tunableop_results.csv"
else
    echo "[TUNING] PyTorch TunableOp disabled (use --tuning to enable)"
    # Still use tuned results if they exist (PyTorch appends GPU index to filename)
    if [ -f "${PROJECT_DIR}/tunableop_results0.csv" ]; then
        echo "         Using cached tuning results from previous run"
        export PYTORCH_TUNABLEOP_ENABLED=1
        export PYTORCH_TUNABLEOP_TUNING=0
        export PYTORCH_TUNABLEOP_FILENAME="${PROJECT_DIR}/tunableop_results.csv"
    elif [ -f "${PROJECT_DIR}/tunableop_results.csv" ]; then
        echo "         Using cached tuning results from previous run"
        export PYTORCH_TUNABLEOP_ENABLED=1
        export PYTORCH_TUNABLEOP_TUNING=0
        export PYTORCH_TUNABLEOP_FILENAME="${PROJECT_DIR}/tunableop_results.csv"
    fi
fi

# Debug settings
if [ "$ENABLE_DEBUG" -eq 1 ]; then
    echo "[DEBUG] Verbose logging ENABLED"
    export TORCH_SHOW_CPP_STACKTRACES=1
    export TORCH_CPP_LOG_LEVEL=INFO
    export AMD_LOG_LEVEL=3
    export HIP_VISIBLE_DEVICES=0
    export HSA_DEBUG=1
else
    echo "[DEBUG] Verbose logging disabled (use --debug to enable)"
fi

echo ""
echo "Environment Configuration:"
echo "  HSA_ENABLE_SDMA=0         (compute shaders instead of DMA)"
echo "  HSA_ENABLE_INTERRUPT=0    (polling instead of interrupts)"
echo "  ROCR_VISIBLE_DEVICES=0    (explicit GPU isolation)"
echo "  VLLM_USE_TRITON_FLASH_ATTN=0 (CK flash attention)"
echo "  TORCH_CUDNN_SDPA_ENABLED=0   (disable unstable SDPA)"
echo "  ROCBLAS_USE_HIPBLASLT=1   (optimized GEMM backend)"
echo ""
echo "=============================================="
echo "Starting experiment..."
echo "=============================================="
echo ""

cd "$PROJECT_DIR"
exec uv run python scripts/experiment_qwen_vl.py
