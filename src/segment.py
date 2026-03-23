"""
SAM2-based segmentation.

Given a PIL image and a list of (x, y) pixel coordinates (one per object),
returns a list of binary numpy masks (H, W, bool), one per point.

Duplicate masks (IoU > 0.85) are removed, keeping the largest one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

# SAM2 imports — installed from facebookresearch/sam2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False


SAM2_MODEL_CFG = "sam2_hiera_large.yaml"
SAM2_CHECKPOINT = "facebook/sam2-hiera-large"   # HF hub id

_IOU_DEDUP_THRESHOLD = 0.85


class SAM2Segmenter:
    def __init__(self, device: Optional[str] = None):
        if not _SAM2_AVAILABLE:
            raise ImportError(
                "sam2 package not found. Install via: "
                "pip install 'git+https://github.com/facebookresearch/sam2.git'"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor: Optional[SAM2ImagePredictor] = None

    def _load(self):
        if self._predictor is not None:
            return
        print(f"[SAM2Segmenter] Loading SAM2 ({SAM2_CHECKPOINT}) ...")
        from sam2.build_sam import build_sam2_hf
        model = build_sam2_hf(SAM2_CHECKPOINT, device=self.device)
        self._predictor = SAM2ImagePredictor(model)
        print("[SAM2Segmenter] Model loaded.")

    def segment(
        self,
        image: Image.Image,
        points: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """
        Segment one object per point.

        Args:
            image: PIL RGB image.
            points: list of (x, y) pixel coordinates (absolute).

        Returns:
            list of bool np.ndarray masks (H, W), one per unique object.
        """
        self._load()

        img_np = np.array(image.convert("RGB"))
        self._predictor.set_image(img_np)

        masks_out = []
        for (x, y) in points:
            pts = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = foreground

            with torch.inference_mode():
                masks, scores, _ = self._predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=True,
                )

            # Pick the mask with the highest score
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(bool)  # (H, W)
            masks_out.append(best_mask)

        return _dedup_masks(masks_out, threshold=_IOU_DEDUP_THRESHOLD)

    def unload(self):
        self._predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union + 1e-6)


def _dedup_masks(masks: list[np.ndarray], threshold: float = 0.85) -> list[np.ndarray]:
    """Remove near-duplicate masks (IoU > threshold), keeping the larger one."""
    kept = []
    for mask in masks:
        duplicate = False
        for i, existing in enumerate(kept):
            if _mask_iou(mask, existing) > threshold:
                # Keep the larger mask
                if mask.sum() > existing.sum():
                    kept[i] = mask
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
    return kept
