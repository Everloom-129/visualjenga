"""
Molmo-based object detection.

Uses allenai/Molmo-7B-D-0924 to point at every distinct object in the image.
Molmo returns coordinates as XML-like tags:
    <point x="0.42" y="0.67">cat</point>

Returns a list of (label, x_frac, y_frac) where x/y are in [0, 1] (fraction of W/H).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


MOLMO_MODEL_ID = "allenai/Molmo-7B-D-0924"

_POINT_PATTERN = re.compile(
    r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"[^>]*>\s*([^<]*?)\s*</point>',
    re.IGNORECASE,
)

_DETECT_PROMPT = (
    "Point to each distinct object in this image. "
    "Include all foreground objects individually."
)


@dataclass
class DetectedObject:
    label: str
    x_frac: float  # [0, 1] fraction of image width
    y_frac: float  # [0, 1] fraction of image height

    def pixel_coords(self, width: int, height: int) -> tuple[int, int]:
        return int(self.x_frac * width), int(self.y_frac * height)


class MolmoDetector:
    def __init__(self, device: Optional[str] = None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        print(f"[MolmoDetector] Loading {MOLMO_MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(
            MOLMO_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MOLMO_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()
        print("[MolmoDetector] Model loaded.")

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Return a list of detected objects with fractional (x, y) coordinates."""
        self._load()

        inputs = self._processor.process(
            images=[image],
            text=_DETECT_PROMPT,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)

        with torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )

        # Decode only the newly generated tokens
        generated_ids = output[0, inputs["input_ids"].size(1):]
        response = self._processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return _parse_points(response)

    def unload(self):
        """Free GPU memory."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_points(text: str) -> list[DetectedObject]:
    """Parse Molmo point tags from generated text."""
    objects = []
    for match in _POINT_PATTERN.finditer(text):
        x_frac = float(match.group(1)) / 100.0  # Molmo outputs 0-100
        y_frac = float(match.group(2)) / 100.0
        label = match.group(3).strip() or "object"
        # Clamp to [0, 1]
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        objects.append(DetectedObject(label=label, x_frac=x_frac, y_frac=y_frac))
    return objects
