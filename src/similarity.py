"""
CLIP + DINO similarity utilities.

Both return cosine similarity normalized to [0, 1]:
    sim_01 = (cosine_sim + 1) / 2

Used by diversity.py to score inpainted crops against the original crop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import open_clip
from transformers import AutoImageProcessor, AutoModel as HFAutoModel


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

_CLIP_MODEL_NAME = "ViT-L-14"
_CLIP_PRETRAINED = "openai"

# ---------------------------------------------------------------------------
# DINO / DINOv2
# ---------------------------------------------------------------------------

_DINO_MODEL_NAME = "facebook/dinov2-base"  # 224×224 input, 768-dim features


class SimilarityModel:
    """
    Loads CLIP (ViT-L/14) and DINOv2-base once and provides per-image similarity.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_preprocess = None
        self._dino_model = None
        self._dino_processor = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_clip(self):
        if self._clip_model is not None:
            return
        print(f"[SimilarityModel] Loading CLIP {_CLIP_MODEL_NAME} ...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        model = model.to(self.device).eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        print("[SimilarityModel] CLIP loaded.")

    def _load_dino(self):
        if self._dino_model is not None:
            return
        print(f"[SimilarityModel] Loading DINOv2 {_DINO_MODEL_NAME} ...")
        self._dino_processor = AutoImageProcessor.from_pretrained(_DINO_MODEL_NAME)
        self._dino_model = HFAutoModel.from_pretrained(_DINO_MODEL_NAME).to(self.device).eval()
        print("[SimilarityModel] DINOv2 loaded.")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def clip_features(self, image: Image.Image) -> torch.Tensor:
        self._load_clip()
        tensor = self._clip_preprocess(image).unsqueeze(0).to(self.device)
        feats = self._clip_model.encode_image(tensor)
        return F.normalize(feats, dim=-1)

    @torch.inference_mode()
    def dino_features(self, image: Image.Image) -> torch.Tensor:
        self._load_dino()
        inputs = self._dino_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._dino_model(**inputs)
        # Use the [CLS] token embedding
        feats = outputs.last_hidden_state[:, 0, :]
        return F.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Similarity (returned in [0, 1])
    # ------------------------------------------------------------------

    def clip_sim(self, img_a: Image.Image, img_b: Image.Image) -> float:
        fa = self.clip_features(img_a)
        fb = self.clip_features(img_b)
        cos = (fa * fb).sum().item()  # already L2-normalised
        return (cos + 1.0) / 2.0

    def dino_sim(self, img_a: Image.Image, img_b: Image.Image) -> float:
        fa = self.dino_features(img_a)
        fb = self.dino_features(img_b)
        cos = (fa * fb).sum().item()
        return (cos + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Batch variants (more efficient when scoring N inpaintings at once)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def clip_sim_batch(
        self, originals: list[Image.Image], inpaintings: list[Image.Image]
    ) -> list[float]:
        """Compute CLIP sim between each original[i] and inpainting[i]."""
        self._load_clip()
        orig_tensors = torch.stack([self._clip_preprocess(i) for i in originals]).to(self.device)
        inp_tensors = torch.stack([self._clip_preprocess(i) for i in inpaintings]).to(self.device)
        f_orig = F.normalize(self._clip_model.encode_image(orig_tensors), dim=-1)
        f_inp = F.normalize(self._clip_model.encode_image(inp_tensors), dim=-1)
        cos = (f_orig * f_inp).sum(dim=-1).cpu().tolist()
        return [(c + 1.0) / 2.0 for c in cos]

    @torch.inference_mode()
    def dino_sim_batch(
        self, originals: list[Image.Image], inpaintings: list[Image.Image]
    ) -> list[float]:
        """Compute DINOv2 sim between each original[i] and inpainting[i]."""
        self._load_dino()
        def _encode(imgs):
            inputs = self._dino_processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self._dino_model(**inputs)
            return F.normalize(out.last_hidden_state[:, 0, :], dim=-1)

        f_orig = _encode(originals)
        f_inp = _encode(inpaintings)
        cos = (f_orig * f_inp).sum(dim=-1).cpu().tolist()
        return [(c + 1.0) / 2.0 for c in cos]

    def unload(self):
        self._clip_model = None
        self._clip_preprocess = None
        self._dino_model = None
        self._dino_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
