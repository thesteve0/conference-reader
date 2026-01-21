"""
SmolLM3 Experimentation Script

This script experiments with SmolLM3-135M for text summarization of conference posters.
It loads pre-serialized ProcessedDocuments to avoid re-running Docling on each iteration.

Usage:
    python scripts/experiment_smollm3.py

Goals:
    1. Test model loading and verify ROCm/GPU acceleration
    2. Experiment with different prompt templates
    3. Measure memory usage and inference speed
    4. Evaluate summary quality
"""

import pickle
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Configuration
SERIALIZED_DOCS_PATH = "output/serialized_document_extractions.pkl"
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


def load_documents():
    """Load pre-serialized ProcessedDocuments."""
    print(f"Loading documents from: {SERIALIZED_DOCS_PATH}")
    with open(SERIALIZED_DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    print(f"✓ Loaded {len(documents)} documents\n")
    return documents


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
    """Load SmolLM3-135M model and tokenizer."""
    print("\n" + "=" * 80)
    print("MODEL LOADING")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")

    start_time = time.time()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Tokenizer loaded")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()  # Set to evaluation mode

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")

    # Check model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Check memory usage after loading
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU memory allocated: {mem_allocated:.2f} GB")
        print(f"  GPU memory reserved: {mem_reserved:.2f} GB")

    return model, tokenizer, device


def create_prompt(title: str, text: str) -> str:
    """
    Prompt Template: Example-driven prompt with specific format.

    Provides an example to guide the model's output format.
    The model receives the full extracted text.

    Args:
        title: The extracted title from the first line of markdown
        text: The full extracted text (sent to model in its entirety)
    """
    prompt = f"""You are helping someone quickly scan conference posters. Summarize the poster below in 1-2 sentences, focusing on the main research contribution and results.

Example:
Title: "CrypticBio: A large multimodal dataset"
Summary: A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given.

Title: {title}

Content:
{text}

Summary:"""
    return prompt


def generate_summary(
    model, tokenizer, device, prompt: str, max_new_tokens: int = 100
) -> str:
    """
    Generate a summary using the model.

    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        device: Device to run on (cuda/cpu)
        prompt: The input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated summary text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for testing
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start_time

    # Decode (skip the input prompt)
    generated_text = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True
    )

    # Calculate tokens per second
    output_length = outputs.shape[1] - input_length
    tokens_per_sec = output_length / gen_time if gen_time > 0 else 0

    return generated_text.strip(), gen_time, tokens_per_sec, input_length, output_length


def experiment_with_prompts(model, tokenizer, device, documents):
    """
    Test the prompt template on the first document.

    Note: Title is extracted from the first line of markdown text
    (the first heading with # markers stripped).
    """
    print("\n" + "=" * 80)
    print("PROMPT EXPERIMENTATION")
    print("=" * 80)

    # Just use first document for quick testing
    doc = documents[0]
    if not doc.success:
        print("First document failed extraction, skipping")
        return

    # Extract title (first line of markdown text)
    lines = doc.extracted_text.split("\n")
    title = lines[0].strip("#").strip() if lines else "Unknown"

    # Truncate text if too long (for initial testing)
    max_chars = 2000
    text = doc.extracted_text[:max_chars]
    if len(doc.extracted_text) > max_chars:
        text += "\n[... truncated]"

    print(f"\nTest Document: {doc.filename}")
    print(f"Extracted title: {title}")
    print(f"Title source: First line of markdown text")
    print(f"Text length: {len(doc.extracted_text)} chars (using {len(text)} chars)")

    # Create prompt
    prompt = create_prompt(title, text)

    print("\n" + "=" * 80)
    print("Example-Driven Prompt Template")
    print("=" * 80)

    # Show extracted text preview
    text_preview = text[:100] + "..." if len(text) > 100 else text
    print("\n[EXTRACTED TEXT - First 100 chars]")
    print("-" * 80)
    print(text_preview)
    print("-" * 80)

    # Show the prompt with truncated content for readability
    # (actual prompt sent to model has full text)
    content_for_display = text[:50] + "..." if len(text) > 50 else text
    prompt_display = f"""You are helping someone quickly scan conference posters. Summarize the poster below in 1-2 sentences, focusing on the main research contribution and results.

Example:
Title: "CrypticBio: A large multimodal dataset"
Summary: A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given.

Title: {title}

Content:
{content_for_display}

Summary:"""

    print("\n[PROMPT SENT TO MODEL - content truncated for display]")
    print("-" * 80)
    print(prompt_display)
    print("-" * 80)

    summary, gen_time, tokens_per_sec, input_len, output_len = generate_summary(
        model, tokenizer, device, prompt, max_new_tokens=80
    )

    print(
        f"\nMetrics: {input_len} input tokens | {output_len} output tokens | {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    print("\n[GENERATED SUMMARY]")
    print("-" * 80)
    print(summary)
    print("-" * 80)


def test_all_documents(model, tokenizer, device, documents):
    """
    Generate summaries for all documents using the example-driven prompt.
    """
    print("\n" + "=" * 80)
    print("BATCH SUMMARIZATION TEST")
    print("=" * 80)
    print("\nUsing example-driven prompt template for all documents")
    print("Prompt shows first 50 chars of content only")
    print("Max tokens: 80\n")

    successful_docs = [doc for doc in documents if doc.success]

    total_time = 0
    for i, doc in enumerate(successful_docs, 1):
        # Extract title (first line of markdown text)
        lines = doc.extracted_text.split("\n")
        title = lines[0].strip("#").strip() if lines else "Unknown"

        # Truncate text
        max_chars = 2000
        text = doc.extracted_text[:max_chars]
        text_preview = text[:100] + "..." if len(text) > 100 else text

        # Generate summary
        prompt = create_prompt(title, text)
        summary, gen_time, tokens_per_sec, _, _ = generate_summary(
            model, tokenizer, device, prompt, max_new_tokens=80
        )

        total_time += gen_time

        print("=" * 80)
        print(f"[{i}/{len(successful_docs)}] {doc.filename}")
        print("=" * 80)
        print(f"\n[TITLE] (from first line of markdown)")
        print(title)
        print(f"\n[EXTRACTED TEXT - First 100 chars]")
        print(text_preview)
        print(f"\n[GENERATED SUMMARY]")
        print(summary)
        print(f"\n[METRICS] {gen_time:.2f}s | {tokens_per_sec:.1f} tok/s")

    avg_time = total_time / len(successful_docs)
    print("\n" + "=" * 80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per document: {avg_time:.2f}s")
    print("=" * 80)


def main():
    """Main experimentation workflow."""
    print("\n" + "=" * 80)
    print("SMOLLM3-3B EXPERIMENTATION")
    print("=" * 80 + "\n")

    # Step 1: Check GPU
    check_gpu()

    # Step 2: Load model
    model, tokenizer, device = load_model()

    # Step 3: Load documents
    documents = load_documents()

    # Step 4: Test all documents
    test_all_documents(model, tokenizer, device, documents)

    print("\n" + "=" * 80)
    print("EXPERIMENTATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
