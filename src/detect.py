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

from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


MOLMO_MODEL_ID = "allenai/Molmo-7B-D-0924"

_POINT_PATTERN = re.compile(
    r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"[^>]*>\s*([^<]*?)\s*</point>',
    re.IGNORECASE,
)
_POINTS_TAG_PATTERN = re.compile(
    r"<points\b([^>]*)>(.*?)</points>",
    re.IGNORECASE | re.DOTALL,
)
_POINTS_TAG_OPEN_PATTERN = re.compile(
    r"<points\b([^>]*)>?(.*)",
    re.IGNORECASE | re.DOTALL,
)
_XY_ATTR_PATTERN = re.compile(
    r'\bx(\d+)\s*=\s*"([0-9.]+)"\s+y\1\s*=\s*"([0-9.]+)"',
    re.IGNORECASE,
)

_DETECT_PROMPT = "Point to every distinct object."

# Maximum character length for a label before we treat it as junk (e.g. echoed prompt text)
_MAX_LABEL_LEN = 60

# Minimum fractional distance between two kept points (NMS in normalised coords).
# Two points closer than this are considered duplicates; the first one is kept.
_NMS_MIN_DIST_FRAC = 0.05


@dataclass
class DetectedObject:
    label: str
    x_frac: float  # [0, 1] fraction of image width
    y_frac: float  # [0, 1] fraction of image height

    def pixel_coords(self, width: int, height: int) -> tuple[int, int]:
        return int(self.x_frac * width), int(self.y_frac * height)


class MolmoDetector:
    def __init__(self, device: Optional[str] = None, dtype=None, debug: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.debug = debug
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
            device_map="auto",
        )
        self._model.eval()
        print("[MolmoDetector] Model loaded.")

    def __del__(self):
        self.unload()

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Return a list of detected objects with fractional (x, y) coordinates."""
        self._load()
        if self.debug:
            print("[MolmoDetector][debug] Starting detection")
            print(f"[MolmoDetector][debug] Prompt: {_DETECT_PROMPT}")
            print(f"[MolmoDetector][debug] Image size: {image.size[0]}x{image.size[1]}")

        raw = self._processor.process(images=[image], text=_DETECT_PROMPT)
        primary_device = next(self._model.parameters()).device
        inputs = {}
        for k, v in raw.items():
            if not hasattr(v, "unsqueeze"):
                inputs[k] = v
                continue
            v = v.unsqueeze(0)
            if v.is_floating_point():
                v = v.to(primary_device, dtype=self.dtype)
            else:
                v = v.to(primary_device)
            inputs[k] = v

        with torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )

        # Decode only the newly generated tokens
        generated_ids = output[0, inputs["input_ids"].size(1):]
        response = self._processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        if self.debug:
            preview = response.replace("\n", "\\n")
            if len(preview) > 600:
                preview = preview[:600] + "...<truncated>"
            print(f"[MolmoDetector][debug] Raw response preview: {preview}")

        parsed = _parse_points(response, debug=self.debug)
        if self.debug:
            print(f"[MolmoDetector][debug] Parsed object count: {len(parsed)}")
        return parsed

    def unload(self):
        """Free GPU memory."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sanitise_label(raw: str, fallback: str) -> str:
    """Return a clean object label, falling back when raw looks like echoed prompt text."""
    label = raw.strip()
    if not label or len(label) > _MAX_LABEL_LEN:
        return fallback
    return label


def _nms_points(objects: list[DetectedObject], min_dist: float) -> list[DetectedObject]:
    """
    Spatial NMS: drop any point whose fractional distance to an earlier kept point
    is less than `min_dist`.  Preserves insertion order (first-seen wins).
    """
    kept: list[DetectedObject] = []
    for obj in objects:
        too_close = any(
            ((obj.x_frac - k.x_frac) ** 2 + (obj.y_frac - k.y_frac) ** 2) ** 0.5 < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(obj)
    return kept


def _parse_points(text: str, debug: bool = False) -> list[DetectedObject]:
    """Parse Molmo point tags from generated text.

    Supported formats:
      1) <point x="42.1" y="67.3">label</point>
      2) <points x1="..." y1="..." x2="..." y2="..." ...>optional label</points>

    After parsing, spatial NMS removes near-duplicate points (e.g. Molmo tracing
    a single object with 40 evenly-spaced coordinates).
    """
    objects = []
    matches = list(_POINT_PATTERN.finditer(text))
    if debug:
        print(f"[MolmoDetector][debug] <point> matches: {len(matches)}")
    for idx, match in enumerate(matches):
        x_frac = float(match.group(1)) / 100.0  # Molmo outputs 0-100
        y_frac = float(match.group(2)) / 100.0
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        label = _sanitise_label(match.group(3), fallback=f"object_{idx + 1}")
        objects.append(DetectedObject(label=label, x_frac=x_frac, y_frac=y_frac))
        if debug:
            print(
                f"[MolmoDetector][debug] Parsed[{idx}]: "
                f"label='{label}', x_frac={x_frac:.4f}, y_frac={y_frac:.4f}"
            )

    # Fallback for Molmo outputs that use a single <points ...> tag with xN/yN attrs.
    if objects:
        return _nms_points(objects, _NMS_MIN_DIST_FRAC)

    # Try the properly-closed tag first, then fall back to an open/truncated tag.
    points_tag_match = _POINTS_TAG_PATTERN.search(text)
    truncated = False
    if points_tag_match is None:
        points_tag_match = _POINTS_TAG_OPEN_PATTERN.search(text)
        truncated = True
    if points_tag_match is None:
        return objects

    attrs = points_tag_match.group(1) or ""
    inner_text = (points_tag_match.group(2) or "").strip()
    if truncated:
        inner_text = ""
    xy_matches = list(_XY_ATTR_PATTERN.finditer(attrs))
    if debug:
        tag_kind = "truncated <points>" if truncated else "<points>"
        print(f"[MolmoDetector][debug] {tag_kind} xN/yN matches found: {len(xy_matches)}")

    # Extract the alt attribute as label if present
    alt_match = re.search(r'alt="([^"]*)"', attrs)
    alt_label = alt_match.group(1).strip() if alt_match else ""

    # inner_text can be the echoed prompt if Molmo misformats the tag — sanitise it
    shared_label = _sanitise_label(inner_text, fallback="") or _sanitise_label(alt_label, fallback="")

    for idx, match in enumerate(xy_matches):
        x_frac = float(match.group(2)) / 100.0
        y_frac = float(match.group(3)) / 100.0
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        label = shared_label or f"object_{idx + 1}"
        objects.append(DetectedObject(label=label, x_frac=x_frac, y_frac=y_frac))
        if debug:
            print(
                f"[MolmoDetector][debug] ParsedFromPoints[{idx}]: "
                f"label='{label}', x_frac={x_frac:.4f}, y_frac={y_frac:.4f}"
            )

    return _nms_points(objects, _NMS_MIN_DIST_FRAC)
