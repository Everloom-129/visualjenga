"""
Diversity score computation (Eq. 2 from the Visual Jenga paper).

For a given object mask, the diversity score measures how much the inpainted
region varies across N samples:

    diversity = 1 - mean_j(ClipSim(c_new_j, c_orig))
                  × mean_j(DinoSim(c_new_j, c_orig))
                  / area_fraction

where:
  - c_orig  = crop of the original image to the mask bounding box,
              224×224, pixels outside the mask are zeroed
  - c_new_j = same crop applied to the j-th inpainting
  - area_fraction = (mask pixels in bounding box) / (bounding box area)

A HIGH diversity score means the object slot can be filled by many different
things → the object is NOT structurally required → remove it first.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .similarity import SimilarityModel


_CROP_SIZE = 224  # CLIP and DINO both expect 224×224


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------

def _tight_square_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the smallest SQUARE bounding box containing mask.

    All indices are exclusive at the end (i.e., mask[y0:y1, x0:x1] contains
    every True pixel in the mask).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # Use exclusive end indices so Python slicing works correctly
    r0 = int(np.where(rows)[0][0])
    r1 = int(np.where(rows)[0][-1]) + 1   # exclusive
    c0 = int(np.where(cols)[0][0])
    c1 = int(np.where(cols)[0][-1]) + 1   # exclusive

    h = r1 - r0
    w = c1 - c0
    side = max(h, w)

    # Centre the square over the tight bbox
    H, W = mask.shape
    cy = (r0 + r1) // 2
    cx = (c0 + c1) // 2

    y0 = max(0, cy - side // 2)
    y1 = min(H, y0 + side)
    if y1 - y0 < side:
        y0 = max(0, y1 - side)

    x0 = max(0, cx - side // 2)
    x1 = min(W, x0 + side)
    if x1 - x0 < side:
        x0 = max(0, x1 - side)

    return int(x0), int(y0), int(x1), int(y1)


def crop_to_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Crop image to the square bounding box of mask, zero pixels outside mask,
    resize to 224×224.

    Args:
        image: PIL RGB image (H, W, 3).
        mask: bool (H, W) mask.

    Returns:
        PIL RGB image (224, 224).
    """
    x0, y0, x1, y1 = _tight_square_bbox(mask)

    img_np = np.array(image.convert("RGB"))
    crop_img = img_np[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    # Zero out pixels outside the mask
    crop_img[~crop_mask] = 0

    pil = Image.fromarray(crop_img)
    return pil.resize((_CROP_SIZE, _CROP_SIZE), Image.BILINEAR)


def area_fraction(mask: np.ndarray) -> float:
    """
    Fraction of the square bounding box covered by the mask.
    Compensates for large bounding boxes with small actual masks.
    """
    x0, y0, x1, y1 = _tight_square_bbox(mask)
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask[y0:y1, x0:x1].sum()) / float(bbox_area)


# ---------------------------------------------------------------------------
# Diversity score
# ---------------------------------------------------------------------------

def diversity_score(
    original: Image.Image,
    mask: np.ndarray,
    inpaintings: list[Image.Image],
    sim_model: SimilarityModel,
) -> float:
    """
    Compute the diversity score for one object.

    Args:
        original:    Original PIL image.
        mask:        Bool (H, W) mask for this object.
        inpaintings: N inpainted PIL images (same size as original).
        sim_model:   Loaded SimilarityModel.

    Returns:
        Scalar diversity score in approximately [0, 1].
        Higher = more diverse inpaintings = object should be removed first.
    """
    orig_crop = crop_to_mask(original, mask)
    inp_crops = [crop_to_mask(inp, mask) for inp in inpaintings]

    n = len(inp_crops)
    orig_repeated = [orig_crop] * n

    clip_sims = sim_model.clip_sim_batch(orig_repeated, inp_crops)
    dino_sims = sim_model.dino_sim_batch(orig_repeated, inp_crops)

    mean_clip = float(np.mean(clip_sims))
    mean_dino = float(np.mean(dino_sims))
    af = area_fraction(mask)

    # Eq. 2: diversity = 1 - (mean_clip × mean_dino) / area_fraction
    # area_fraction normalisation upweights small objects
    score = 1.0 - (mean_clip * mean_dino) / max(af, 1e-6)
    return score
