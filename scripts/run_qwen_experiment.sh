#!/bin/bash
# Wrapper script to test different ROCm configurations for Qwen3-VL-4B

echo "================================================================================"
echo "ATTEMPTING OPTION 4: Experimental ROCm Features"
echo "================================================================================"
echo ""
echo "Setting TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
echo "This enables experimental attention optimizations on AMD GPUs"
echo ""

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Activate virtual environment
source /workspaces/conference-reader/.venv/bin/activate

# Run the experiment
python scripts/experiment_qwen_vl.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ SUCCESS: Option 4 (Experimental ROCm) worked!"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "✗ FAILED: Option 4 (Experimental ROCm) did not work"
    echo "Exit code: $exit_code"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "1. Try Option 2: Eager attention with float16"
    echo "2. Try Option 3: Float32 on GPU"
    echo "3. Fallback to Option 1: CPU only"
fi

exit $exit_code
