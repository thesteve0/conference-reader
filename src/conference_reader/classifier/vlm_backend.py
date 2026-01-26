"""VLM Backend wrapper for Qwen3-VL with ROCm stability settings.

This module provides a clean interface for loading and running inference
with the Qwen3-VL-4B vision-language model, optimized for ROCm/AMD GPUs.
"""

import time
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


DeviceMode = Literal[
    "eager_float16",  # Stable on Strix Halo
    "eager_bfloat16",  # Recommended for ROCm 7.2+
    "eager_float32",  # Slower, more memory
    "cpu",  # Fallback
]


class VLMBackend:
    """Wrapper for Qwen3-VL-4B with ROCm stability settings.

    This class handles model loading and inference with proper configuration
    for stable execution on AMD GPUs.

    Attributes:
        model: The loaded Qwen3-VL model
        processor: The AutoProcessor for tokenization and image processing
        device: The device the model is loaded on

    Example:
        >>> backend = VLMBackend()
        >>> response = backend.generate(Path("poster.jpg"), "Describe this image")
        >>> print(response)
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

    # Image resolution limits (controls token count and memory usage)
    # Qwen VL uses 28x28 pixel patches
    DEFAULT_MIN_PIXELS = 256 * 28 * 28  # ~200K pixels minimum
    DEFAULT_MAX_PIXELS = 512 * 28 * 28  # ~400K pixels maximum

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device_mode: DeviceMode = "eager_float16",
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ):
        """Load model with proper ROCm configuration.

        Args:
            model_name: HuggingFace model identifier
            device_mode: GPU/CPU mode for inference:
                - "eager_float16": Stable on Strix Halo (default)
                - "eager_bfloat16": Recommended for ROCm 7.2+
                - "eager_float32": Slower, more memory
                - "cpu": Fallback, slowest
            min_pixels: Minimum image resolution in pixels
            max_pixels: Maximum image resolution in pixels
        """
        self.model_name = model_name
        self.device_mode = device_mode

        # Configure model loading based on device mode
        load_kwargs = self._get_load_kwargs(device_mode)

        # Load processor with image resolution limits
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        )

        self.device = str(self.model.device)

    def _get_load_kwargs(self, device_mode: DeviceMode) -> dict:
        """Get model loading kwargs for the specified device mode.

        Args:
            device_mode: The device mode to configure

        Returns:
            Dictionary of kwargs for from_pretrained()

        Raises:
            ValueError: If device_mode is not recognized
        """
        if device_mode == "eager_float16":
            return {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "attn_implementation": "eager",
            }
        elif device_mode == "eager_bfloat16":
            return {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "attn_implementation": "eager",
            }
        elif device_mode == "eager_float32":
            return {
                "torch_dtype": torch.float32,
                "device_map": "auto",
                "attn_implementation": "eager",
            }
        elif device_mode == "cpu":
            return {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "attn_implementation": "eager",
            }
        else:
            raise ValueError(f"Unknown device_mode: {device_mode}")

    def generate(
        self,
        image_path: Path,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, float]:
        """Run inference on an image with a prompt.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, inference_time_seconds)
        """
        # Load image
        image = Image.open(image_path)

        # Create messages in Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        inference_time = time.time() - start_time

        # Decode output (trim input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else "", inference_time

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False
