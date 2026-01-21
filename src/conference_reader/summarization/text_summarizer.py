"""TextSummarizer for generating summaries using SmolLM3-3B."""

from dataclasses import replace
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..extraction import ProcessedDocument


class TextSummarizer:
    """Generates concise summaries of extracted text using SmolLM3-3B.

    This class loads the SmolLM3-3B model once during initialization and
    generates 1-2 sentence summaries for conference posters. It uses an
    example-driven prompt template to guide the model's output format.

    The class follows the immutable pattern: it returns new ProcessedDocument
    instances with the summary field populated, leaving the originals unchanged.

    Usage:
        summarizer = TextSummarizer()
        documents_with_summaries = summarizer.summarize_batch(documents)
    """

    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
    MAX_NEW_TOKENS = 80
    MAX_INPUT_CHARS = 4000  # Truncate long texts

    def __init__(self):
        """Initialize the summarizer and load the SmolLM3-3B model.

        The model is loaded once during initialization and cached for
        subsequent summarization calls. Uses GPU (CUDA) if available,
        otherwise falls back to CPU.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

    def _extract_title(self, doc: ProcessedDocument) -> str:
        """Extract title from the first line of markdown text.

        Args:
            doc: ProcessedDocument with extracted_text

        Returns:
            Title string with # symbols stripped, or "Unknown" if not found
        """
        lines = doc.extracted_text.split("\n")
        if lines:
            return lines[0].strip("#").strip()
        return "Unknown"

    def _create_prompt(self, title: str, text: str) -> str:
        """Create the example-driven prompt for summarization.

        Args:
            title: The extracted title (from first markdown line)
            text: The full extracted text (will be truncated if too long)

        Returns:
            Formatted prompt string
        """
        # Truncate text if too long
        truncated_text = text[: self.MAX_INPUT_CHARS]
        if len(text) > self.MAX_INPUT_CHARS:
            truncated_text += "\n[... truncated]"

        prompt = f"""You are helping someone quickly scan conference posters. Summarize the poster below in 1-2 sentences, focusing on the main research contribution and results.

Example:
Title: "CrypticBio: A large multimodal dataset"
Summary: A dataset was created and tested for building and testing models' ability to detect differences in similar species. Comparisons of different model results are given.

DO NOT repeat this example as your response. You MUST generate a new summary based on the provided poster text which follows `Content:`.

Title: {title}

Content:
{truncated_text}

Summary:"""
        return prompt

    def _generate_summary(self, prompt: str) -> str:
        """Generate summary using the model.

        Args:
            prompt: The formatted prompt string

        Returns:
            Generated summary text (cleaned and stripped)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,  # Deterministic generation
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (skip the input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        return generated_text.strip()

    def summarize_single(self, doc: ProcessedDocument) -> ProcessedDocument:
        """Generate summary for a single document.

        Args:
            doc: ProcessedDocument with extracted text

        Returns:
            New ProcessedDocument with summary field populated.
            If summarization fails or doc.success is False, returns
            doc with summary=None.
        """
        # Skip if extraction failed
        if not doc.success or not doc.extracted_text:
            return doc

        try:
            # Extract title and create prompt
            title = self._extract_title(doc)
            prompt = self._create_prompt(title, doc.extracted_text)

            # Generate summary
            summary = self._generate_summary(prompt)

            # Return new document with summary
            return replace(doc, summary=summary)

        except Exception as e:
            # If summarization fails, return doc with summary=None
            # Could log the error here if needed
            print(f"Warning: Summarization failed for {doc.filename}: {e}")
            return replace(doc, summary=None)

    def summarize_batch(
        self, documents: List[ProcessedDocument]
    ) -> List[ProcessedDocument]:
        """Generate summaries for a batch of documents.

        Args:
            documents: List of ProcessedDocument instances

        Returns:
            List of new ProcessedDocument instances with summaries populated.
            Documents that failed extraction will have summary=None.
        """
        return [self.summarize_single(doc) for doc in documents]
