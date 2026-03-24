"""
Stable Diffusion 1.5 inpainting.

Uses runwayml/stable-diffusion-inpainting to fill masked regions.

Two modes:
  - inpaint(image, mask, n): N diverse samples for diversity scoring.
  - remove(image, mask): single high-quality removal for the final output.

The mask is dilated before inpainting to avoid edge artifacts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline


SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

_POSITIVE_PROMPT = (
    "Full HD, 4K, high quality, high resolution, photorealistic"
)
_NEGATIVE_PROMPT = (
    "bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, "
    "duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, "
    "low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, "
    "ugly, worst quality"
)

_MASK_DILATION_PX = 10
_INFER_STEPS = 50
_GUIDANCE_SCALE = 7.5


class SDInpainter:
    def __init__(self, device: Optional[str] = None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        self._pipe: Optional[StableDiffusionInpaintPipeline] = None

    def _load(self):
        if self._pipe is not None:
            return
        print(f"[SDInpainter] Loading {SD_INPAINT_MODEL} ...")
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        print("[SDInpainter] Model loaded.")

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        n: int = 16,
    ) -> list[Image.Image]:
        """
        Generate N diverse inpaintings of the masked region.

        Args:
            image: Original PIL RGB image.
            mask: bool or uint8 (H, W) mask — True/1 = region to inpaint.
            n: Number of samples to generate.

        Returns:
            List of N PIL images (same size as input).
        """
        self._load()
        dilated_mask = _dilate_mask(mask, _MASK_DILATION_PX)
        mask_pil = _mask_to_pil(dilated_mask, image.size)
        img_rgb = image.convert("RGB")

        results = []
        for seed in range(n):
            generator = torch.Generator(device=self.device).manual_seed(seed)
            out = self._pipe(
                prompt=_POSITIVE_PROMPT,
                negative_prompt=_NEGATIVE_PROMPT,
                image=img_rgb,
                mask_image=mask_pil,
                num_inference_steps=_INFER_STEPS,
                guidance_scale=_GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
            results.append(out.resize(image.size, Image.LANCZOS))
        return results

    def remove(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        num_inference_steps: int = 75,
    ) -> Image.Image:
        """
        Single high-quality removal for the Visual Jenga output sequence.
        Uses more inference steps for better quality.
        """
        self._load()
        dilated_mask = _dilate_mask(mask, _MASK_DILATION_PX)
        mask_pil = _mask_to_pil(dilated_mask, image.size)
        img_rgb = image.convert("RGB")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipe(
            prompt=_POSITIVE_PROMPT,
            negative_prompt=_NEGATIVE_PROMPT,
            image=img_rgb,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=_GUIDANCE_SCALE,
            generator=generator,
        ).images[0]
        return result.resize(image.size, Image.LANCZOS)

    def unload(self):
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Morphologically dilate binary mask by `pixels`."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1)
    )
    m = mask.astype(np.uint8)
    return cv2.dilate(m, kernel).astype(bool)


def _mask_to_pil(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Convert bool (H, W) mask to white-on-black PIL image matching `size` (W, H)."""
    m = (mask.astype(np.uint8) * 255)
    pil = Image.fromarray(m, mode="L")
    if pil.size != size:
        pil = pil.resize(size, Image.NEAREST)
    return pil
