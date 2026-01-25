"""Qwen3-VL-4B Experimentation Script

Tests Qwen3-VL-4B for classifying conference poster images.
Distinguishes between full posters and QR code crops.

Usage:
    python scripts/experiment_qwen_vl.py

Goals:
    1. Test VLM loading and GPU acceleration
    2. Evaluate image classification accuracy
    3. Measure inference time and memory usage
    4. Display full model reasoning and output

ROCm 7.2 Configuration:
    Default: "eager_bfloat16" - Recommended for ROCm 7.2 with optimized GEMM kernels

    Environment variables already set in devcontainer.json:
    - ROCBLAS_USE_HIPBLASLT=1: Enables hipBLASLt backend (~35 TFLOPS vs ~5 TFLOPS)
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: Improves memory management

    To switch between different modes, change the DEVICE_MODE variable below:
    - "eager_bfloat16": GPU with eager attention + bfloat16 (RECOMMENDED for ROCm 7.2)
    - "eager_float16": GPU with eager attention + float16
    - "eager_float32": GPU with eager attention + float32 (slower, more memory)
    - "auto": Try GPU with experimental features (may be unstable)
    - "cpu": Force CPU execution (slowest, most reliable)
"""

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================================
# CONFIGURATION - Change DEVICE_MODE to switch between GPU workarounds
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# Change this to try different GPU/CPU modes:
# "eager_bfloat16" = RECOMMENDED: ROCm 7.2 optimized with Origami GEMM tuning, 2x throughput vs FP32
# "eager_float16" = Alternative: GPU with eager attention + float16
# "eager_float32" = Legacy: GPU with eager attention + float32 (slower, more memory)
# "auto" = Experimental: Try GPU with experimental features (may be unstable)
# "cpu" = Fallback: Force CPU (most reliable, slowest)
DEVICE_MODE = (
    "eager_float16"  # STABLE: bf16 hangs on Strix Halo, float16 works
)

# Image evaluation configuration
IMAGE_DIR = Path("/data/neurips/invalid_poster_images")
GROUND_TRUTH_FILE = Path("datasets/eval/validator_ground_truth.json")

# Image resolution limits (controls token count and memory usage)
# Qwen VL uses 28x28 pixel patches, so max_pixels = width * 28 * 28
# Lower values = fewer tokens = less memory but potentially lower accuracy
# Default would be ~2048*28*28 = 1,605,632 which creates too many tokens
MIN_PIXELS = 256 * 28 * 28      # ~200K pixels minimum
MAX_PIXELS = 512 * 28 * 28      # ~400K pixels maximum (reduced for memory)
# ============================================================================

CLASSIFICATION_PROMPT = """Analyze this image and determine if it shows:
A) A FULL conference poster - You can see the complete poster with:
   - A clear title at the top
   - Multiple sections of content
   - The full layout and structure
   - Sometimes there might be people blocking part of the poster
   - If you can see multiple borders of the poster, it is likely a full poster

B) A CROPPED SECTION - You only see:
   - A zoomed-in portion of a poster
   - Mainly QR codes and very little other content
   - Missing the title and main sections
   - Just a small fragment, not the whole poster

Respond with ONLY ONE WORD:
- Answer "poster" if this is a FULL, complete conference poster
- Answer "qr" if this is a CROPPED section showing mainly QR codes

Your one-word answer:"""


def load_ground_truth(ground_truth_file: Path) -> dict[str, str]:
    """Load ground truth from JSON file and convert to poster/qr labels.

    Args:
        ground_truth_file: Path to JSON file with ground truth labels

    Returns:
        Dictionary mapping filenames to "poster" or "qr" classifications

    Notes:
        - JSON format: true = valid full poster, false = invalid (QR/crop)
        - Converts to: "poster" for true, "qr" for false
    """
    with open(ground_truth_file, "r") as f:
        data = json.load(f)

    # Convert true/false to poster/qr
    ground_truth = {}
    for filename, is_valid_poster in data.items():
        if filename.startswith("_"):  # Skip comment fields
            continue
        if is_valid_poster is None:  # Skip unlabeled
            continue
        ground_truth[filename] = "poster" if is_valid_poster else "qr"

    return ground_truth


def get_image_paths(image_dir: Path, ground_truth: dict[str, str]) -> list[Path]:
    """Get all image paths from directory that have ground truth labels.

    Args:
        image_dir: Directory containing images
        ground_truth: Dictionary of ground truth labels

    Returns:
        List of image paths sorted by filename
    """
    image_paths = []
    for filename in sorted(ground_truth.keys()):
        image_path = image_dir / filename
        if image_path.exists():
            image_paths.append(image_path)
        else:
            print(f"⚠ Warning: Image not found: {image_path}")

    return image_paths


def check_gpu():
    """Check GPU availability and print device info."""
    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(0)}")

        # Check memory
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Memory allocated: {mem_allocated:.2f} GB")
        print(f"  Memory reserved: {mem_reserved:.2f} GB")
        return True
    else:
        print("✗ CUDA not available - will use CPU")
        return False


def load_model():
    """Load Qwen3-VL-4B model and processor."""
    print("\n" + "=" * 80)
    print("MODEL LOADING")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device Mode: {DEVICE_MODE}")

    # Configure model loading based on DEVICE_MODE
    if DEVICE_MODE == "auto":
        # Option 4: Try GPU with experimental ROCm (requires env var set externally)
        print("\n🔬 Mode: Experimental ROCm (auto dtype/device)")
        print("   Requires: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1")
        load_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }
    elif DEVICE_MODE == "eager_bfloat16":
        # RECOMMENDED for ROCm 7.2: Optimized GEMM kernels + 2x throughput vs FP32
        print("\n🚀 Mode: Eager attention + bfloat16 on GPU (ROCm 7.2 Optimized)")
        print(
            "   Benefits: Origami-tuned kernels, 2x memory bandwidth, native gfx1151 support"
        )
        print(
            "   Performance: ~35 TFLOPS with hipBLASLt vs ~5 TFLOPS on legacy backend"
        )
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
    elif DEVICE_MODE == "eager_float16":
        # Option 2: GPU with eager attention + float16
        print("\n🔧 Mode: Eager attention + float16 on GPU")
        print("   Strategy: Disable SDPA, use float16")
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
    elif DEVICE_MODE == "eager_float32":
        # Option 3: GPU with eager attention + float32
        print("\n🔧 Mode: Eager attention + float32 on GPU")
        print("   Strategy: Disable SDPA, use float32 (more memory)")
        load_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
    elif DEVICE_MODE == "cpu":
        # Option 1: Force CPU
        print("\n💻 Mode: CPU only (no GPU)")
        print("   Strategy: Most reliable, slowest")
        load_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "attn_implementation": "eager",
        }
    else:
        raise ValueError(f"Unknown DEVICE_MODE: {DEVICE_MODE}")

    start_time = time.time()

    # Load processor with image resolution limits
    print("\nLoading processor...")
    print(f"  Image resolution: min={MIN_PIXELS:,} max={MAX_PIXELS:,} pixels")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print("✓ Processor loaded")

    # Load model with configured settings
    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, **load_kwargs)

    # Optional: Enable torch.compile for better performance on ROCm 7.2 + PyTorch 2.9.1
    # Reduces CPU-launch latency (~60s → ~15s cold start)
    # Uncomment to enable:
    # if DEVICE_MODE != "cpu":
    #     print("\nCompiling model with torch.compile (reduce-overhead mode)...")
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("✓ Model compiled")

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")

    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count/1e9:.1f}B)")

    # Check which device was selected
    device = str(model.device)
    print(f"  Device: {device}")

    # If on CUDA, show memory usage
    if torch.cuda.is_available() and "cuda" in device:
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU memory allocated: {mem_allocated:.2f} GB")
        print(f"  GPU memory reserved: {mem_reserved:.2f} GB")

    return model, processor


def classify_image(model, processor, image_path: str, prompt: str):
    """Classify a single image using Qwen3-VL-4B.

    Args:
        model: Loaded VLM model
        processor: AutoProcessor for Qwen3-VL
        image_path: Path to image file
        prompt: Classification prompt

    Returns:
        tuple: (classification, full_output, gen_time, tokens_per_sec,
                input_length, output_length)
    """
    print(f"  [Stage 1/4] Loading image...")
    stage_start = time.time()
    image = Image.open(image_path)
    print(f"  ✓ Image loaded ({time.time() - stage_start:.2f}s)")

    # Create messages in Qwen3-VL format
    print(f"  [Stage 2/4] Processing inputs...")
    stage_start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # CORRECT: Single-step processing with tokenize=True
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    input_length = inputs.input_ids.shape[1]
    print(f"  ✓ Inputs processed ({time.time() - stage_start:.2f}s)")
    print(f"    Input tokens: {input_length}")

    # Generate with increased max_new_tokens to see full reasoning
    print(f"  [Stage 3/4] Running model inference...")
    print(f"    ⚠ First inference may take 5-6 minutes due to:")
    print(f"       - JIT compilation / kernel compilation")
    print(f"       - GPU kernel caching")
    print(f"       - Attention pattern optimization")
    print(f"    ⚠ Subsequent inferences should be much faster (~10-20s)")

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # Reduced from 512: sufficient for classification, improves stability
            do_sample=False,
        )
    gen_time = time.time() - start_time
    print(f"  ✓ Inference complete ({gen_time:.2f}s)")

    # Calculate tokens/sec
    output_length = generated_ids.shape[1] - input_length
    tokens_per_sec = output_length / gen_time if gen_time > 0 else 0
    print(f"    Generated {output_length} tokens at {tokens_per_sec:.1f} tokens/sec")

    # CORRECT: Trim input tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    full_output = output_text[0] if output_text else ""

    # Extract classification from full output
    output_lower = full_output.lower()
    if "poster" in output_lower and "qr" not in output_lower:
        classification = "poster"
    elif "qr" in output_lower:
        classification = "qr"
    else:
        classification = "unclear"

    # Calculate metrics
    output_length = len(generated_ids_trimmed[0])
    tokens_per_sec = output_length / gen_time if gen_time > 0 else 0

    return (
        classification,
        full_output,
        gen_time,
        tokens_per_sec,
        input_length,
        output_length,
    )


def test_all_images(model, processor, image_paths, ground_truth, prompt):
    """Test classification on all images.

    Args:
        model: Loaded VLM model
        processor: AutoProcessor for the model
        image_paths: List of image paths to classify
        ground_truth: Dictionary mapping filenames to expected classifications
        prompt: Classification prompt to use
    """
    print("\n" + "=" * 80)
    print("IMAGE CLASSIFICATION TEST")
    print("=" * 80)
    print(f"Testing {len(image_paths)} images with ground truth validation")

    results = []
    total_time = 0
    correct_count = 0

    for i, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).name
        print(f"\n[{i}/{len(image_paths)}] {image_path}")
        print("-" * 80)

        # Classify image
        (
            classification,
            full_output,
            gen_time,
            tokens_per_sec,
            input_len,
            output_len,
        ) = classify_image(model, processor, image_path, prompt)

        total_time += gen_time

        # Check against ground truth
        expected = ground_truth.get(filename, "unknown")
        is_correct = classification == expected
        if is_correct:
            correct_count += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"

        # Print full model output
        print(f"\n[FULL MODEL OUTPUT]")
        print("-" * 80)
        print(full_output)
        print("-" * 80)

        # Print results
        print(f"\nExtracted Classification: {classification}")
        print(f"Expected: {expected}")
        print(f"Result: {status}")
        print(f"Inference time: {gen_time:.2f}s")
        print(f"Tokens/sec: {tokens_per_sec:.1f}")
        print(f"Input tokens: {input_len}, Output tokens: {output_len}")

        results.append(
            {
                "filename": filename,
                "classification": classification,
                "expected": expected,
                "correct": is_correct,
                "time": gen_time,
            }
        )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(image_paths)}")
    accuracy = (correct_count / len(image_paths) * 100) if image_paths else 0
    print(
        f"Correct classifications: {correct_count}/{len(image_paths)} ({accuracy:.1f}%)"
    )
    avg_time = total_time / len(image_paths) if image_paths else 0
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    return results


def main():
    """Main experimentation workflow."""
    print("\n" + "=" * 80)
    print("QWEN3-VL-4B EXPERIMENTATION")
    print("=" * 80 + "\n")

    # Step 1: Load ground truth and get image paths
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    print(f"Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"Image directory: {IMAGE_DIR}")

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"✓ Loaded {len(ground_truth)} ground truth labels")

    image_paths = get_image_paths(IMAGE_DIR, ground_truth)
    print(f"✓ Found {len(image_paths)} images to test\n")

    # Step 2: Check GPU
    check_gpu()

    # Step 3: Load model
    model, processor = load_model()

    # Step 4: Test all images
    test_all_images(model, processor, image_paths, ground_truth, CLASSIFICATION_PROMPT)

    print("\n" + "=" * 80)
    print("EXPERIMENTATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
