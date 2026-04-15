"""
Visual Jenga end-to-end pipeline.

Implements the full iterative object-removal loop:
    1. Detect objects (Molmo)           ┐ Phase A — unload before Phase B
    2. Segment each object (SAM2)       ┘
    3. Score each object (SD inpainting + CLIP/DINO diversity)  ┐ Phase B
    4. Remove the highest-diversity object (SD inpainting)      ┘
    5. Repeat from step 1 on the updated image

Memory management (A10 / 24 GB):
    Molmo-7B alone takes ~14 GB bfloat16.  SAM2 + SD + CLIP + DINO together
    take another ~5 GB.  To avoid OOM the pipeline explicitly:
      • loads Molmo → detects → unloads Molmo (and empties cache)
      • loads SAM2  → segments → unloads SAM2 (and empties cache)
      • loads SD + CLIP + DINO → scores + removes → unloads all (and empties cache)

Intermediate outputs (for debugging / visualisation):
    output_dir/
      step_00_original.png
      step_NN/
        original.png          — current image entering this step
        detect.json           — raw Molmo output (label, x_frac, y_frac)
        detect_viz.png        — image with coloured detection dots
        mask_OO_<label>.png   — binary mask for each object (255 = object)
        mask_OO_<label>_viz.png — mask overlay (red tint) on current image
        scores.json           — diversity score for every object
        inpaint_OO_SS.png     — (optional) every SD sample used for scoring
        pre_removal.png       — current image annotated with chosen object (yellow dot)
        removed_<label>.png   — image after removal (= input to next step)
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch

from .detect import MolmoDetector, DetectedObject
from .segment import SAM2Segmenter
from .inpaint import SDInpainter, LaMaRemover
from .similarity import SimilarityModel
from .diversity import diversity_score


# Palette for detection dot colours (cycles if > 8 objects)
_DOT_PALETTE = [
    "#FF4040", "#40FF40", "#4040FF", "#FFD700",
    "#FF40FF", "#40FFFF", "#FF8000", "#FFFFFF",
]


@dataclass
class ScoredObject:
    detected: DetectedObject
    mask: np.ndarray       # bool (H, W)
    score: float           # diversity score — higher = remove first


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class   VisualJengaPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        n_samples: int = 16,
        max_steps: Optional[int] = None,
        verbose: bool = True,
        save_inpaint_samples: bool = False,
        remover: str = "lama",
    ):
        """
        Args:
            device:               CUDA device string, e.g. "cuda:0". Defaults to auto.
            n_samples:            Number of inpainting samples per object for diversity scoring.
            max_steps:            Maximum number of objects to remove (None = until empty).
            verbose:              Print progress messages.
            save_inpaint_samples: If True, save every SD sample to disk (disk-heavy).
            remover:              Backend for the final removal step: "lama" (default) or "sd".
                                  SD1.5 scores and LaMa removes by default; use "sd" to use
                                  SD1.5 for both (matches original behaviour).
        """
        if remover not in ("lama", "sd"):
            raise ValueError(f"remover must be 'lama' or 'sd', got {remover!r}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_samples = n_samples
        self.max_steps = max_steps
        self.verbose = verbose
        self.save_inpaint_samples = save_inpaint_samples
        self.remover = remover

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        output_dir: str,
        on_step_complete: Optional[Callable[[int, list, list, str], None]] = None,
    ) -> list[Image.Image]:
        """
        Run Visual Jenga on `image`, saving frames and debug logs to `output_dir`.

        Args:
            image:            Input PIL image.
            output_dir:       Directory to save all outputs.
            on_step_complete: Optional callback called after each successful removal.
                              Signature: (step, detected: list[DetectedObject],
                                          scored: list[ScoredObject], step_dir: str) -> None
                              `scored` is sorted descending by score; scored[0] is what was removed.

        Returns:
            List of PIL images: [original, after step 1, after step 2, …]
        """
        os.makedirs(output_dir, exist_ok=True)

        current = image.convert("RGB")
        frames: list[Image.Image] = [current.copy()]
        _save(current, os.path.join(output_dir, "step_00_original.png"))
        self._log(f"Output dir: {output_dir}")

        step = 1
        while True:
            if self.max_steps is not None and step > self.max_steps:
                break

            self._log(f"\n{'='*60}")
            self._log(f"=== Step {step} ===")
            self._log(f"{'='*60}")

            step_dir = os.path.join(output_dir, f"step_{step:02d}")
            os.makedirs(step_dir, exist_ok=True)

            # Save a copy of the current image entering this step
            _save(current, os.path.join(step_dir, "original.png"))

            # ── Phase A: Detect + Segment ─────────────────────────────
            detected, masks = self._phase_a_detect_and_segment(
                current, step, step_dir
            )
            if not detected:
                self._log("  No objects detected — done.")
                break

            # ── Phase B: Score + Remove ───────────────────────────────
            result, scored = self._phase_b_score_and_remove(
                current, detected, masks, step, step_dir
            )
            if result is None:
                self._log("  All masks empty — done.")
                break

            current = result
            frames.append(current.copy())
            if on_step_complete is not None:
                on_step_complete(step, detected, scored, step_dir)
            step += 1

        self._log(f"\nDone — {step - 1} objects removed. Frames saved to {output_dir}")
        return frames

    # ------------------------------------------------------------------
    # Phase A: Detect (Molmo) → unload → Segment (SAM2) → unload
    # ------------------------------------------------------------------

    def _phase_a_detect_and_segment(
        self,
        image: Image.Image,
        step: int,
        step_dir: str,
    ) -> tuple[list[DetectedObject], list[np.ndarray]]:

        # ── 1. Detect ──────────────────────────────────────────────────
        self._log(f"  [Phase A] Step {step}: Loading Molmo ...")
        detector = MolmoDetector(device=self.device, debug=self.verbose)
        detected = detector.detect(image)
        self._log(f"  [Phase A] Detected {len(detected)} objects: {[d.label for d in detected]}")
        detector.unload()
        del detector
        _free_gpu()
        self._log(f"  [Phase A] Molmo unloaded. GPU memory freed.")

        # Save detection results
        _save_json(
            [{"label": d.label, "x_frac": d.x_frac, "y_frac": d.y_frac} for d in detected],
            os.path.join(step_dir, "detect.json"),
        )
        _save(
            _draw_detections(image, detected),
            os.path.join(step_dir, "detect_viz.png"),
        )
        self._log(f"  [Phase A] Saved detect.json + detect_viz.png")

        if not detected:
            return [], []

        # ── 2. Segment ─────────────────────────────────────────────────
        self._log("  [Phase A] Loading SAM2 ...")
        segmenter = SAM2Segmenter(device=self.device)
        W, H = image.size
        points = [d.pixel_coords(W, H) for d in detected]
        masks = segmenter.segment(image, points)
        self._log(f"  [Phase A] Got {len(masks)} masks (after dedup).")
        segmenter.unload()
        del segmenter
        _free_gpu()
        self._log(f"  [Phase A] SAM2 unloaded. GPU memory freed.")

        # Align detected to masks (dedup may have shortened list)
        if len(masks) < len(detected):
            detected = detected[:len(masks)]

        # Save each mask + overlay
        for obj_idx, (det, mask) in enumerate(zip(detected, masks)):
            safe_label = _safe_filename(det.label)
            mask_uint8 = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8, mode="L").save(
                os.path.join(step_dir, f"mask_{obj_idx:02d}_{safe_label}.png")
            )
            _save(
                _draw_mask_overlay(image, mask, color=(220, 50, 50, 120)),
                os.path.join(step_dir, f"mask_{obj_idx:02d}_{safe_label}_viz.png"),
            )
        self._log(f"  [Phase A] Saved {len(masks)} mask images.")

        return detected, masks

    # ------------------------------------------------------------------
    # Phase B: Score (SD + CLIP + DINO) → Remove (SD) → unload all
    # ------------------------------------------------------------------

    def _phase_b_score_and_remove(
        self,
        image: Image.Image,
        detected: list[DetectedObject],
        masks: list[np.ndarray],
        step: int,
        step_dir: str,
    ) -> tuple[Optional[Image.Image], list[ScoredObject]]:

        self._log(f"  [Phase B] Step {step}: Loading SD Inpainter + CLIP + DINOv2 ...")
        inpainter = SDInpainter(device=self.device)
        sim_model = SimilarityModel(device=self.device)
        # Trigger lazy loads now so we can report memory once
        inpainter._load()
        sim_model._load_clip()
        sim_model._load_dino()
        self._log("  [Phase B] All Phase B models loaded.")

        # ── 3. Score each object ───────────────────────────────────────
        self._log(f"  [Phase B] Scoring {len(masks)} objects (n_samples={self.n_samples}) ...")
        scored: list[ScoredObject] = []
        scores_log: list[dict] = []

        for obj_idx, (det, mask) in enumerate(zip(detected, masks)):
            if mask.sum() == 0:
                self._log(f"    [{obj_idx}] {det.label}: mask empty — skip")
                continue

            inpaintings = inpainter.inpaint(image, mask, n=self.n_samples)

            # Optionally save SD samples
            if self.save_inpaint_samples:
                for s_idx, inp in enumerate(inpaintings):
                    safe_label = _safe_filename(det.label)
                    _save(inp, os.path.join(step_dir, f"inpaint_{obj_idx:02d}_{s_idx:02d}_{safe_label}.png"))

            score = diversity_score(image, mask, inpaintings, sim_model)
            scored.append(ScoredObject(detected=det, mask=mask, score=score))
            scores_log.append({"obj_idx": obj_idx, "label": det.label, "score": score})
            self._log(f"    [{obj_idx}] {det.label:25s}  diversity={score:.4f}")

        _save_json(scores_log, os.path.join(step_dir, "scores.json"))
        self._log(f"  [Phase B] Saved scores.json")

        if not scored:
            inpainter.unload()
            sim_model.unload()
            del inpainter, sim_model
            _free_gpu()
            return None, []

        # ── 4. Select object with highest diversity ────────────────────
        scored.sort(key=lambda s: s.score, reverse=True)
        target = scored[0]
        self._log(f"  [Phase B] → Removing '{target.detected.label}' (score={target.score:.4f})")

        # Save annotated pre-removal frame
        viz = _annotate_target(image, target)
        _save(viz, os.path.join(step_dir, "pre_removal.png"))

        # ── 5. Remove ──────────────────────────────────────────────────
        safe_label = _safe_filename(target.detected.label)
        if self.remover == "lama":
            # Unload SD + similarity before loading LaMa to stay within GPU budget
            inpainter.unload()
            sim_model.unload()
            del inpainter, sim_model
            _free_gpu()
            self._log("  [Phase B] SD + CLIP + DINO unloaded. Loading LaMa for removal ...")
            remover_obj = LaMaRemover(device=self.device)
            result = remover_obj.remove(image, target.mask)
            remover_obj.unload()
            del remover_obj
            _free_gpu()
            self._log("  [Phase B] LaMa unloaded. GPU memory freed.")
        else:
            # SD removal — inpainter is still loaded, use it directly
            result = inpainter.remove(image, target.mask)
            inpainter.unload()
            sim_model.unload()
            del inpainter, sim_model
            _free_gpu()
            self._log("  [Phase B] SD + CLIP + DINO unloaded. GPU memory freed.")

        _save(result, os.path.join(step_dir, f"removed_{safe_label}.png"))
        return result, scored

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_detections(image: Image.Image, detected: list[DetectedObject]) -> Image.Image:
    """Overlay coloured dots + labels for each detected object."""
    W, H = image.size
    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    r = max(8, min(W, H) // 60)
    for idx, det in enumerate(detected):
        color = _DOT_PALETTE[idx % len(_DOT_PALETTE)]
        x, y = det.pixel_coords(W, H)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline="black", width=2)
        draw.text((x + r + 2, y - r), det.label, fill=color)
    return img.convert("RGB")


def _draw_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int, int] = (220, 50, 50, 120),
) -> Image.Image:
    """Tint the masked region with a semi-transparent colour."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pixels = np.array(overlay)
    pixels[mask] = color
    overlay = Image.fromarray(pixels, mode="RGBA")
    composite = Image.alpha_composite(base, overlay)
    return composite.convert("RGB")


def _annotate_target(image: Image.Image, target: ScoredObject) -> Image.Image:
    """Draw a yellow dot on the object that is about to be removed."""
    W, H = image.size
    x, y = target.detected.pixel_coords(W, H)
    img = image.copy()
    draw = ImageDraw.Draw(img)
    r = max(8, min(W, H) // 60)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="yellow", outline="black", width=2)
    draw.text((x + r + 2, y - r), f"{target.detected.label} ({target.score:.3f})", fill="yellow")
    return img


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _safe_filename(label: str, max_len: int = 40) -> str:
    """Convert an object label to a safe filename fragment."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return safe[:max_len]


def _save(image: Image.Image, path: str):
    image.save(path)


def _save_json(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
