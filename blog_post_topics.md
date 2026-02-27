# Blog Post Topics from Conference Poster Summarization Project

This document captures potential blog post topics and lessons learned from building a VLM-based pipeline for extracting and summarizing conference poster images.

## Technical Deep Dives

### 1. Running VLMs on AMD GPUs: A ROCm Survival Guide

- The challenges of getting Qwen3-VL and other models working on ROCm/Strix Halo
- Environment variables that matter: `HSA_ENABLE_SDMA=0`, `TORCH_CUDNN_SDPA_ENABLED=0`, etc.
- Why you need `attn_implementation="eager"` and can't use flash attention (yet)
- The tunable ops caching system for performance
- Disabling cuDNN/MIOpen for stability

### 2. A Practical Pipeline for Conference Poster Summarization with Local VLMs

- The full architecture: classify -> OCR -> summarize
- Why you need a classification step (QR codes vs posters)
- Choosing the right model for each stage (Qwen3-VL for vision, SmolLM3 for text)
- Memory management when chaining multiple models
- Exporting structured results to CSV

### 3. OCR Engine Shootout: EasyOCR vs RapidOCR vs Tesseract for Conference Posters

- We tried multiple OCR approaches during development
- RapidOCR failures on certain images
- Why EasyOCR won out for this use case
- The tradeoffs between pure VLM extraction vs dedicated OCR
- Image scaling strategies for better OCR accuracy

## Lessons Learned

### 4. Why VLM-Only Poster Reading Doesn't Work (Yet)

- The initial dream: feed poster image directly to VLM, get summary
- Reality: VLMs hallucinate titles and produce generic summaries
- The hybrid approach: OCR for text extraction + LLM for summarization
- When VLMs shine (classification) vs where they struggle (dense OCR)
- Token limits when processing large amounts of extracted text

### 5. Managing GPU Memory When Chaining AI Models

- Loading/unloading models sequentially (`classifier.unload()`)
- The `_reset_reader()` pattern after slow extractions (>60s threshold)
- `torch.cuda.empty_cache()` and garbage collection between models
- Why you can't just load 3 models simultaneously on consumer hardware
- Detecting and recovering from GPU resource exhaustion

### 6. Prompt Engineering for Poster Summarization: Fighting Example Parroting

- Why few-shot prompting can backfire with smaller models
- Explicitly telling the model "DO NOT repeat this example"
- Token limits and truncation strategies (`MAX_INPUT_CHARS = 4000`)
- Balancing prompt length with output quality
- Deterministic generation (`do_sample=False`) for reproducibility

## Feasibility Assessment

### The Honest Take on VLM-Based Poster Summarization

#### What Worked Well

- **Image classification (poster vs QR)** - Near 100% accuracy with simple prompting
- **The pipeline architecture** - Clean separation of concerns between classification, extraction, and summarization
- **Local inference on consumer AMD GPU** - No API costs, full privacy
- **EasyOCR text extraction** - Reliable and fast enough for batch processing

#### What Was Harder Than Expected

- **VLMs struggle with dense, structured text extraction** - Conference posters have complex layouts that confuse pure vision models
- **ROCm compatibility required significant tuning** - Many environment variables and attention implementation changes needed
- **Memory management across multiple models is non-trivial** - Had to implement explicit unload/reload patterns
- **Small VLMs produce generic summaries** - Without strong prompting constraints, outputs were too vague
- **Some images cause OCR hangs** - Needed timeout handling and reader reset logic

#### Verdict

**Feasible but hybrid approach wins.** Use VLMs for what they're good at (understanding visual layout, classification) and dedicated OCR + text LLMs for extraction and summarization. The three-stage pipeline (classify -> extract -> summarize) proved more reliable than attempting end-to-end VLM processing.

## Project Architecture Reference

```
conference-reader/
├── main.py                           # Full pipeline orchestration
├── src/conference_reader/
│   ├── classifier/
│   │   ├── image_classifier.py       # Qwen3-VL poster vs QR classification
│   │   └── vlm_backend.py            # ROCm-optimized VLM wrapper
│   ├── extraction/
│   │   └── document_extractor.py     # EasyOCR text extraction
│   ├── summarization/
│   │   └── text_summarizer.py        # SmolLM3 text summarization
│   └── config/
│       └── rocm_config.py            # AMD GPU stability settings
```

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| Classification | Qwen3-VL-4B-Instruct | Distinguish posters from QR code crops |
| Text Extraction | EasyOCR | Extract text from poster images |
| Summarization | SmolLM3-3B | Generate 1-2 sentence summaries |

## Key Configuration Insights

### ROCm Stability Settings

```python
settings = {
    "HSA_ENABLE_SDMA": "0",
    "HSA_ENABLE_INTERRUPT": "0",
    "ROCR_VISIBLE_DEVICES": "0",
    "VLLM_USE_TRITON_FLASH_ATTN": "0",
    "TORCH_CUDNN_SDPA_ENABLED": "0",
    "ROCBLAS_USE_HIPBLASLT": "1",
    "PYTORCH_TUNABLEOP_ENABLED": "1",
}
```

### VLM Loading for ROCm

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",  # Required for ROCm stability
)
```

---

# Lessons Learned for Future Projects

## Commit Early, Commit Often: Working with AI Assistants

### Why This Matters

When working with Claude Code (or any AI assistant), things will sometimes go wrong. The AI might:
- Refactor something in a way you didn't want
- Break working code while "improving" it
- Go down an approach that doesn't pan out
- Make changes across multiple files that are hard to untangle

**Git commits are your safety net.** Small, frequent commits at logical stopping points make it trivial to back out when needed.

### The Pattern

1. **Commit before starting a new feature or experiment**
   ```bash
   git add . && git commit -m "Working baseline before adding summarization"
   ```

2. **Commit after each working increment**
   - OCR extraction works? Commit.
   - Classification works? Commit.
   - Single file refactor done? Commit.

3. **Commit before risky changes**
   ```bash
   git commit -m "Before refactoring extractor to use EasyOCR"
   ```

4. **Use descriptive messages that capture state**
   - Good: "Works but titles were garbage and I want to be able to see QR versus Poster classification"
   - Good: "Workflow works but we have problems with some images failing in RapidOCR"
   - These messages help you find good rollback points later

### Recovery Patterns

```bash
# See recent commits to find a good state
git log --oneline -10

# Soft reset to undo commits but keep changes
git reset --soft HEAD~2

# Hard reset to completely restore a known good state
git reset --hard abc123

# Create a branch to preserve current work before resetting
git branch experiment-that-failed
git reset --hard last-known-good
```

### Real Examples from This Project

Looking at the commit history:
- `"just before running on full collection"` - Captured state before a big batch run
- `"Works but titles were garbage..."` - Honest about partial success
- `"figured out how to filter qr versus poster"` - Marked a feature completion

Each of these was a safe point to return to when later experiments didn't work out.

### Working with Claude Code Specifically

- **Ask Claude to commit after successful changes** - "This works, let's commit before moving on"
- **Don't let changes accumulate** - Multiple uncommitted file changes are hard to untangle
- **Use `git diff` to review** - Before committing, see what actually changed
- **Be explicit about rollback** - "Let's go back to the last commit and try a different approach"

### Automate the Reminder: Add This to CLAUDE.md

To have Claude proactively remind you about commits, add this to your project's `CLAUDE.md` file:

```markdown
## Git Commit Reminders

**REQUIRED**: After completing a working feature, fixing a bug, or reaching a logical stopping point, ask the user: "This seems like a good stopping point. Would you like me to create a git commit?"

Good stopping points include:
- A new feature works end-to-end
- A bug is fixed and verified
- A refactor is complete and tests pass
- Before starting a risky or experimental change
- After adding a new file or module

This helps ensure frequent commits so changes can be easily rolled back if needed.
```

This makes commit discipline automatic rather than something you have to remember.

## Working with AMD GPUs, ROCm, and PyTorch

### The Good

- **ROCm has come a long way** - Most PyTorch operations work out of the box
- **Consumer AMD GPUs are viable for inference** - Strix Halo and similar chips can run 4B+ parameter models
- **No API costs** - Local inference means unlimited experimentation
- **HIP is mostly CUDA-compatible** - Most code "just works" after the initial setup

### The Challenges

- **Environment variables are critical** - You must set them BEFORE importing torch:
  ```python
  import os
  os.environ["HSA_ENABLE_SDMA"] = "0"  # Before torch import!
  import torch
  ```
- **Flash attention doesn't work** - Use `attn_implementation="eager"` for all models
- **cuDNN/MIOpen can cause hangs** - Disable with `torch.backends.cudnn.enabled = False`
- **Not all models are tested on ROCm** - Expect to debug; HuggingFace issues are your friend
- **Error messages are cryptic** - "MIOpen workspace allocation failed" often means try different env vars
- **Tunable ops help performance** - Enable `PYTORCH_TUNABLEOP_ENABLED=1` and cache results

### Debugging Tips

- Start with CPU to verify the code works, then add GPU
- Check `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` first
- Monitor GPU memory: `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- When things hang, it's usually attention or convolution ops - try eager mode
- The ROCm GitHub issues and PyTorch forums have most answers

## Working with Claude Code on Iterative Development

### What Worked Well

- **Starting with exploration before coding** - Reading existing code and understanding the codebase first prevents wasted effort
- **Incremental changes** - Small, testable changes are easier to debug than big rewrites
- **Explaining intent clearly** - "I want to filter QR codes from posters" is better than "make the classifier work"
- **Using CLAUDE.md for persistent instructions** - Rules like "use `uv add` not `pip install`" carry across sessions
- **Asking for plans before implementation** - Getting alignment on approach saves rework
- **Serializing intermediate results** - Pickle files let you skip slow OCR when experimenting with summarization

### Tips for Effective Collaboration

- **Be specific about constraints** - "Must work on ROCm" or "Keep memory under 8GB"
- **Share error messages in full** - Don't truncate; the answer is often in the details
- **Say when something didn't work** - "The VLM hallucinated the title" helps me adjust
- **Iterate on prompts together** - Prompt engineering benefits from back-and-forth
- **Let me run experiments** - Scripts like `experiment_smollm3.py` help isolate variables
- **Point out when I'm overcomplicating** - Sometimes simple is better

### Anti-patterns to Avoid

- Don't ask for everything at once - break it into phases
- Don't skip reading existing code - I need context
- Don't hide failures - debugging requires full error output
- Don't expect perfection on the first try - ML is empirical

## Working in Devcontainers

### Benefits We Saw

- **Reproducible environment** - Anyone can rebuild and get the same setup
- **GPU passthrough works** - ROCm/CUDA access through Docker
- **Isolation from host** - No pollution of system Python
- **Easy dependency management** - Dockerfile + devcontainer.json captures everything

### Lessons Learned

- **Mount data volumes** - Large datasets shouldn't live in the container (`/data/neurips/posters`)
- **Cache model weights** - Mount `.cache/huggingface` to persist downloads across rebuilds
- **GPU setup is tricky** - Needed specific `--device` flags and group membership
- **Rebuild when deps change significantly** - Sometimes easier than debugging in-place
- **Use workspace folder mounts** - Code changes persist without rebuilds

### Configuration Tips

```json
// devcontainer.json patterns that helped
{
  "mounts": [
    "source=/data,target=/data,type=bind",
    "source=${localEnv:HOME}/.cache/huggingface,target=/root/.cache/huggingface,type=bind"
  ],
  "runArgs": ["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video"]
}
```

## When to Use Docling vs Straight OCR

### Use Docling When

- You need **structured document understanding** (tables, sections, headers)
- The document has **clear layout hierarchy** you want to preserve
- You're processing **PDFs or multi-page documents**
- You need **markdown or structured output** formats
- Layout detection would **help** downstream processing

### Use Straight OCR (EasyOCR/Tesseract) When

- You just need **raw text extraction**
- Documents are **single images** (like photos of posters)
- Layout is **irregular or artistic** (conference posters often are)
- **Speed matters** more than structure
- Docling's layout model is **getting confused** by the content

### Our Experience

Docling's layout detection was actually counterproductive for conference posters because:
- Posters have non-standard layouts that confused the model
- The overhead wasn't worth it for "just get me the text"
- EasyOCR directly on images was faster and more reliable
- We didn't need structured sections - just text for summarization

## Python Package Management

### Use `uv` Over `pip`

- **Faster** - Significantly faster dependency resolution
- **Reproducible** - Better lock file handling
- **Compatible** - Works with existing pyproject.toml/requirements.txt

### Patterns That Helped

```bash
# Add a package
uv add easyocr

# Add dev dependency
uv add --dev pytest

# Sync environment from lock file
uv sync

# Run a script with proper environment
uv run python main.py
```

### Dependency Lessons

- **Pin major versions** in pyproject.toml, let lock file handle the rest
- **Separate prod and dev deps** - Don't install pytest in production
- **Watch for conflicts** - VLM libraries often have specific torch version requirements
- **Document system deps** - Some packages need `apt install` (tesseract-ocr, etc.)

## VLM-Specific Lessons

### Model Selection

| Task | Model Size Sweet Spot | Why |
|------|----------------------|-----|
| Classification | 3-4B | Simple yes/no, doesn't need reasoning |
| OCR | Dedicated engine | VLMs hallucinate text; OCR doesn't |
| Summarization | 3B+ | Needs enough capacity to understand context |

### Memory Management

- **Unload between stages** - Don't keep classifier loaded while running OCR
- **Use context managers** - `with ImageClassifier() as clf:` ensures cleanup
- **Clear cache explicitly** - `torch.cuda.empty_cache()` after unloading
- **Monitor during development** - Watch `nvidia-smi` or `rocm-smi` during runs

### Prompt Engineering

- **Be explicit about format** - "Respond with ONLY ONE WORD: poster or qr"
- **Show examples but warn against copying** - "DO NOT repeat this example"
- **Truncate long inputs** - Models degrade with very long contexts
- **Use deterministic generation** - `do_sample=False` for reproducibility

## Pipeline Architecture Patterns

### Separation of Concerns Won

```
classify (VLM) -> extract (OCR) -> summarize (LLM)
```

Each stage:
- Has a single responsibility
- Can be tested independently
- Can be replaced without affecting others
- Has clear input/output contracts

### Serialization for Experimentation

```python
# Save intermediate results
with open("extracted_docs.pkl", "wb") as f:
    pickle.dump(documents, f)

# Reload without re-running OCR
with open("extracted_docs.pkl", "rb") as f:
    documents = pickle.load(f)
```

This let us iterate on summarization prompts without waiting for OCR each time.

### Error Recovery Patterns

- **Reset after failures** - The `_reset_reader()` pattern recovers from bad states
- **Timeout thresholds** - Reset after abnormally slow operations (>60s)
- **Graceful degradation** - Return partial results rather than crashing
- **Verbose logging** - Print which file is being processed so you know where it failed
