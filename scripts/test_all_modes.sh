#!/bin/bash
# Comprehensive test script for Qwen3-VL-4B with different ROCm workarounds
# Tests in order: Option 4 → Option 2 → Option 3 → Option 1

set -e  # Exit on error

SCRIPT_DIR="/workspaces/conference-reader"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Function to test a specific mode
test_mode() {
    local mode=$1
    local description=$2
    local extra_env=$3

    echo ""
    echo "================================================================================"
    echo "TESTING: $description"
    echo "Mode: $mode"
    echo "================================================================================"
    echo ""

    # Modify the DEVICE_MODE in the script
    sed -i "s/^DEVICE_MODE = .*/DEVICE_MODE = \"$mode\"  # Auto-set by test script/" scripts/experiment_qwen_vl.py

    # Set environment variables if needed
    if [ -n "$extra_env" ]; then
        export $extra_env
        echo "Environment: $extra_env"
        echo ""
    fi

    # Run the experiment
    if python scripts/experiment_qwen_vl.py; then
        echo ""
        echo "================================================================================"
        echo "✓✓✓ SUCCESS: $description WORKED! ✓✓✓"
        echo "================================================================================"
        echo ""
        echo "The working configuration is: DEVICE_MODE = \"$mode\""
        if [ -n "$extra_env" ]; then
            echo "With environment: $extra_env"
        fi
        echo ""
        echo "You can use this configuration for future runs."
        return 0
    else
        echo ""
        echo "================================================================================"
        echo "✗ FAILED: $description did not work"
        echo "================================================================================"
        return 1
    fi
}

# Save original DEVICE_MODE
original_mode=$(grep "^DEVICE_MODE = " scripts/experiment_qwen_vl.py)

echo "================================================================================"
echo "QWEN3-VL-4B ROCm COMPATIBILITY TEST SUITE"
echo "================================================================================"
echo ""
echo "This script will test 4 different configurations in order:"
echo "  1. Option 4: Experimental ROCm features (fastest if works)"
echo "  2. Option 2: Eager attention + float16 (good compromise)"
echo "  3. Option 3: Eager attention + float32 (more memory)"
echo "  4. Option 1: CPU only (slowest but most reliable)"
echo ""
echo "Testing will stop at the first successful configuration."
echo ""
read -p "Press Enter to start testing..."

# Test Option 4: Experimental ROCm
if test_mode "auto" "Option 4: Experimental ROCm" "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"; then
    exit 0
fi

echo ""
echo "Option 4 failed. Trying Option 2..."
sleep 2

# Test Option 2: Eager + float16
if test_mode "eager_float16" "Option 2: Eager attention + float16"; then
    exit 0
fi

echo ""
echo "Option 2 failed. Trying Option 3..."
sleep 2

# Test Option 3: Eager + float32
if test_mode "eager_float32" "Option 3: Eager attention + float32"; then
    exit 0
fi

echo ""
echo "Option 3 failed. Falling back to Option 1 (CPU)..."
sleep 2

# Test Option 1: CPU (should always work)
if test_mode "cpu" "Option 1: CPU execution"; then
    exit 0
fi

echo ""
echo "================================================================================"
echo "✗✗✗ ALL OPTIONS FAILED ✗✗✗"
echo "================================================================================"
echo ""
echo "This is unexpected. Even CPU mode failed."
echo "Please check the error messages above for details."
echo ""

# Restore original DEVICE_MODE
sed -i "s/^DEVICE_MODE = .*/$(echo $original_mode | sed 's/[\/&]/\\&/g')/" scripts/experiment_qwen_vl.py

exit 1
