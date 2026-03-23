"""
Visual Jenga end-to-end pipeline.

Implements the full iterative object-removal loop:
    1. Detect objects (Molmo)
    2. Segment each object (SAM2)
    3. Score each object (counterfactual inpainting + CLIP/DINO diversity)
    4. Remove the highest-diversity object (SD inpainting)
    5. Repeat from step 1 on the updated image

Each step saves a frame to output_dir as step_NN.png.
The removed object is marked with a yellow dot in the saved visualisation frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .detect import MolmoDetector, DetectedObject
from .segment import SAM2Segmenter
from .inpaint import SDInpainter
from .similarity import SimilarityModel
from .diversity import diversity_score


@dataclass
class ScoredObject:
    detected: DetectedObject
    mask: np.ndarray       # bool (H, W)
    score: float           # diversity score — higher = remove first


class VisualJengaPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        n_samples: int = 16,
        max_steps: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            device:    CUDA device string, e.g. "cuda:0". Defaults to auto.
            n_samples: Number of inpainting samples per object for diversity scoring.
            max_steps: Maximum number of objects to remove (None = until empty).
            verbose:   Print progress messages.
        """
        self.device = device
        self.n_samples = n_samples
        self.max_steps = max_steps
        self.verbose = verbose

        self._detector = MolmoDetector(device=device)
        self._segmenter = SAM2Segmenter(device=device)
        self._inpainter = SDInpainter(device=device)
        self._sim_model = SimilarityModel(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        output_dir: str,
    ) -> list[Image.Image]:
        """
        Run Visual Jenga on `image`, saving frames to `output_dir`.

        Returns:
            List of PIL images: [original, after step 1, after step 2, …]
        """
        os.makedirs(output_dir, exist_ok=True)

        current = image.convert("RGB")
        frames: list[Image.Image] = [current.copy()]
        _save(current, os.path.join(output_dir, "step_00_original.png"))

        step = 1
        while True:
            if self.max_steps is not None and step > self.max_steps:
                break

            self._log(f"\n=== Step {step} ===")

            # 1. Detect
            self._log("  Detecting objects ...")
            detected = self._detector.detect(current)
            if not detected:
                self._log("  No objects detected — done.")
                break
            self._log(f"  Found {len(detected)} objects: {[d.label for d in detected]}")

            # 2. Segment
            self._log("  Segmenting ...")
            W, H = current.size
            points = [d.pixel_coords(W, H) for d in detected]
            masks = self._segmenter.segment(current, points)

            if len(masks) < len(detected):
                # Align masks to detected — zip to shortest
                detected = detected[:len(masks)]

            # 3. Score each object
            self._log(f"  Scoring {len(masks)} objects (n_samples={self.n_samples}) ...")
            scored: list[ScoredObject] = []
            for i, (det, mask) in enumerate(zip(detected, masks)):
                if mask.sum() == 0:
                    continue
                inpaintings = self._inpainter.inpaint(current, mask, n=self.n_samples)
                score = diversity_score(current, mask, inpaintings, self._sim_model)
                scored.append(ScoredObject(detected=det, mask=mask, score=score))
                self._log(f"    [{i}] {det.label:20s}  diversity={score:.4f}")

            if not scored:
                self._log("  All masks empty — done.")
                break

            # 4. Select object with highest diversity
            scored.sort(key=lambda s: s.score, reverse=True)
            target = scored[0]
            self._log(f"  → Removing '{target.detected.label}' (score={target.score:.4f})")

            # Save annotated frame (yellow dot on chosen object)
            viz = _annotate(current, target)
            _save(viz, os.path.join(output_dir, f"step_{step:02d}_pre_removal.png"))

            # 5. Remove
            current = self._inpainter.remove(current, target.mask)
            frames.append(current.copy())
            _save(current, os.path.join(output_dir, f"step_{step:02d}_removed_{target.detected.label}.png"))

            step += 1

        self._log(f"\nDone — {step - 1} objects removed. Frames saved to {output_dir}")
        return frames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _annotate(image: Image.Image, target: ScoredObject) -> Image.Image:
    """Draw a yellow dot on the object that is about to be removed."""
    W, H = image.size
    x, y = target.detected.pixel_coords(W, H)
    img = image.copy()
    draw = ImageDraw.Draw(img)
    r = max(8, min(W, H) // 60)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="yellow", outline="black", width=2)
    return img


def _save(image: Image.Image, path: str):
    image.save(path)
