"""
Inpainting backends.

SDInpainter  — Stable Diffusion 1.5 (runwayml/stable-diffusion-inpainting)
               Used for generating N diverse samples for diversity *scoring*.
               Works on a 512×512 crop around the mask bbox and pastes back.

LaMaRemover  — LaMa (Large Mask inpainting) via simple-lama-inpainting.
               Used for the final object *removal* step that produces the next
               frame.  LaMa operates at full resolution, is trained for clean
               background reconstruction, and does not hallucinate replacement
               objects the way SD1.5 does.

Dilation:
  Both backends dilate the mask before inpainting. Scoring crops (diversity.py)
  always use the original undilated mask.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline
from simple_lama_inpainting import SimpleLama


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
_SD_SIZE = 512          # SD1.5 native resolution
_INPAINT_PAD = 32       # extra context pixels around mask bbox


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

        Uses crop-based inpainting: crops to the mask bounding box, resizes
        to 512×512, runs SD, then pastes the result back into the full image.
        The returned images are the same size as the input.

        Args:
            image: Original PIL RGB image.
            mask: bool or uint8 (H, W) mask — True/1 = region to inpaint.
            n: Number of samples to generate.

        Returns:
            List of N PIL images (same size as input).
        """
        self._load()
        img_rgb = image.convert("RGB")
        img_w, img_h = img_rgb.size

        bbox, img_sd, mask_sd, crop_w, crop_h = _prepare_crop(
            img_rgb, mask, img_w, img_h
        )
        x0, y0 = bbox[0], bbox[1]

        results = []
        for seed in range(n):
            generator = torch.Generator(device=self.device).manual_seed(seed)
            out = self._pipe(
                prompt=_POSITIVE_PROMPT,
                negative_prompt=_NEGATIVE_PROMPT,
                image=img_sd,
                mask_image=mask_sd,
                num_inference_steps=_INFER_STEPS,
                guidance_scale=_GUIDANCE_SCALE,
                generator=generator,
            ).images[0]

            full = _paste_back(img_rgb, out, x0, y0, crop_w, crop_h)
            results.append(full)

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
        Returns a full-resolution image with the object removed.
        """
        self._load()
        img_rgb = image.convert("RGB")
        img_w, img_h = img_rgb.size

        bbox, img_sd, mask_sd, crop_w, crop_h = _prepare_crop(
            img_rgb, mask, img_w, img_h
        )
        x0, y0 = bbox[0], bbox[1]

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self._pipe(
            prompt=_POSITIVE_PROMPT,
            negative_prompt=_NEGATIVE_PROMPT,
            image=img_sd,
            mask_image=mask_sd,
            num_inference_steps=num_inference_steps,
            guidance_scale=_GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

        return _paste_back(img_rgb, out, x0, y0, crop_w, crop_h)

    def unload(self):
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# LaMa-based removal
# ---------------------------------------------------------------------------

class LaMaRemover:
    """
    Object removal using LaMa (Large Mask inpainting).

    Operates at full resolution — no crop/resize needed.  Much cleaner than
    SD1.5 for removal because LaMa is trained to reconstruct backgrounds
    rather than hallucinate replacement objects.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SimpleLama] = None

    def _load(self):
        if self._model is not None:
            return
        print("[LaMaRemover] Loading LaMa model ...")
        self._model = SimpleLama(device=torch.device(self.device))
        print("[LaMaRemover] Model loaded.")

    def remove(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Remove the masked region from `image` using LaMa.

        Args:
            image: Original PIL RGB image.
            mask:  bool or uint8 (H, W) — True/255 = region to remove.

        Returns:
            Full-resolution PIL image with the object replaced by background.
        """
        self._load()
        img_rgb = image.convert("RGB")
        dilated = _dilate_mask(mask, _MASK_DILATION_PX)
        mask_pil = Image.fromarray((dilated.astype(np.uint8) * 255), mode="L")
        result = self._model(img_rgb, mask_pil)
        # SimpleLama may pad to mod-8; crop back to original size if needed
        w, h = img_rgb.size
        if result.size != (w, h):
            result = result.crop((0, 0, w, h))
        return result

    def unload(self):
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Crop-based inpainting helpers (SD1.5)
# ---------------------------------------------------------------------------

def _compute_inpaint_bbox(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    pad: int,
) -> tuple[int, int, int, int]:
    """
    Compute a padded square bounding box for crop-based inpainting.

    Finds the tight bbox of the mask, expands it by `pad` pixels on each
    side (to give SD context beyond the dilation boundary), then makes it
    square and clamps to the image bounds.

    Returns (x0, y0, x1, y1) — all inclusive/exclusive pixel indices.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.where(rows)[0][0])
    r1 = int(np.where(rows)[0][-1]) + 1   # exclusive
    c0 = int(np.where(cols)[0][0])
    c1 = int(np.where(cols)[0][-1]) + 1   # exclusive

    # Expand by pad (ensures dilation fits inside crop + gives context)
    r0 = max(0, r0 - pad)
    r1 = min(img_h, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(img_w, c1 + pad)

    # Make square
    h, w = r1 - r0, c1 - c0
    side = max(h, w)
    cy = (r0 + r1) // 2
    cx = (c0 + c1) // 2

    y0 = max(0, cy - side // 2)
    y1 = min(img_h, y0 + side)
    if y1 - y0 < side:
        y0 = max(0, y1 - side)

    x0 = max(0, cx - side // 2)
    x1 = min(img_w, x0 + side)
    if x1 - x0 < side:
        x0 = max(0, x1 - side)

    return int(x0), int(y0), int(x1), int(y1)


def _prepare_crop(
    img_rgb: Image.Image,
    mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[tuple[int, int, int, int], Image.Image, Image.Image, int, int]:
    """
    Prepare the 512×512 crop and mask for SD inpainting.

    Returns:
        bbox:      (x0, y0, x1, y1) of the crop in the original image
        img_sd:    512×512 PIL crop
        mask_sd:   512×512 PIL dilated mask
        crop_w:    original crop width (for paste-back)
        crop_h:    original crop height (for paste-back)
    """
    pad = _MASK_DILATION_PX + _INPAINT_PAD
    bbox = _compute_inpaint_bbox(mask, img_w, img_h, pad)
    x0, y0, x1, y1 = bbox
    crop_w, crop_h = x1 - x0, y1 - y0

    img_crop = img_rgb.crop((x0, y0, x1, y1))
    mask_crop = mask[y0:y1, x0:x1]

    dilated_crop = _dilate_mask(mask_crop, _MASK_DILATION_PX)

    img_sd = img_crop.resize((_SD_SIZE, _SD_SIZE), Image.LANCZOS)
    mask_sd = _mask_to_pil(dilated_crop, size=(_SD_SIZE, _SD_SIZE))

    return bbox, img_sd, mask_sd, crop_w, crop_h


def _paste_back(
    original: Image.Image,
    inpainted_sd: Image.Image,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
) -> Image.Image:
    """
    Resize `inpainted_sd` (512×512 SD output) back to the original crop
    dimensions and paste into a copy of `original`.

    Returns a full-resolution PIL image.
    """
    out_crop = inpainted_sd.resize((crop_w, crop_h), Image.LANCZOS)
    result = original.copy()
    result.paste(out_crop, (x0, y0))
    return result


# ---------------------------------------------------------------------------
# Low-level mask utilities
# ---------------------------------------------------------------------------

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
