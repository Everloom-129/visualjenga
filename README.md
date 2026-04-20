# Visual Jenga — Reimplementation
- 03/23/2026
- [Jie Wang @ GRASP Lab, UPenn](https://everloom-129.github.io/)

A clean Python reimplementation of **"Visual Jenga: Discovering Object Dependencies via Counterfactual Inpainting"** (Bhattad et al., arXiv 2503.21770).

---

## What is Visual Jenga?

Visual Jenga is a scene understanding task: given a single image, progressively remove objects one at a time—in physically plausible order—until only the background remains. Like the game Jenga, you must understand which objects depend on others before removing them.

The method works by measuring **diversity**: how varied are N inpaintings of the region left behind after masking an object? High diversity means the slot can be filled by anything (the object is not structurally needed) → remove it first. Low diversity means inpainting consistently produces the same thing (something depends on this object being there) → remove it last.

---

## Method Overview

```
Input image
    │
    ▼
[1] Detect objects      — Molmo VLM points at every distinct object
    │
    ▼
[2] Segment             — SAM2 turns each point into a binary mask
    │                     (cached across steps; only new/changed points re-segmented)
    ▼
[3] Score each object   — For each mask:
    │                       • Inpaint the masked region N times (SD1.5)
    │                       • Crop each result to 224×224 (zero outside mask)
    │                       • Compute CLIP + DINO similarity vs. original crop
    │                       • diversity = 1 − mean(ClipSim)×mean(DinoSim) / area_frac
    ▼
[4] Select & remove     — Highest diversity object → LaMa inpainting (final removal)
    │
    ▼
[5] Repeat              — Re-run on updated image until background only (max 10 steps)
```

**Diversity score (Eq. 2 from paper):**

```
diversity(A) = 1 − [ (1/N Σ ClipSim(c_new_j, c_orig)) × (1/N Σ DinoSim(c_new_j, c_orig)) ] / area_fraction
```

---

## Project Structure

```
visualjenga/
├── run_jenga.py              # Single-image CLI entry point
├── run_all_datasets.py       # Batch runner across datasets, with W&B logging
├── dashboard.py              # Streamlit result browser
├── download_dataset.py       # Download evaluation datasets from HuggingFace
├── process_coco.sh           # Convenience script: run full COCO dataset
├── open_dashboard.sh         # Launch Streamlit dashboard
├── pyproject.toml            # Dependencies (uv)
├── src/
│   ├── detect.py             # Molmo-7B object detection → (x, y) points
│   ├── segment.py            # SAM2 segmentation from point prompts
│   ├── inpaint.py            # SD1.5 (scoring) + LaMa (removal) inpainting
│   ├── similarity.py         # CLIP (ViT-L/14) + DINOv2 cosine similarity
│   ├── diversity.py          # Diversity score computation
│   └── pipeline.py           # Full Visual Jenga loop
└── data/
    └── datasets/             # Downloaded evaluation data (via download_dataset.py)
```

---

## Installation

```bash
# Requires Python 3.10+, uv
cd visualjenga
uv sync
```

---

## Usage

### Run on a single image

```bash
uv run python run_jenga.py \
    --image path/to/scene.jpg \
    --output results/ \
    --n 16              # inpainting samples per object for diversity scoring
    --steps 10          # max objects to remove (default: 10)
    --remover lama      # removal backend: lama (default) | sd
    --device cuda:0
```

Output frames are saved as:
```
results/
├── step_00_original.png
├── step_01/
│   ├── original.png
│   ├── detect.json / detect_viz.png
│   ├── mask_00_<label>.png / mask_00_<label>_viz.png
│   ├── scores.json
│   ├── pre_removal.png          # yellow dot marks selected object
│   └── removed_<label>.png     # image after removal
├── step_02/
└── ...
```

### Run the full COCO dataset

```bash
# Edit MODEL= at the top (lama | sd), then:
bash process_coco.sh
```

This runs all 201 COCO scenes and logs results to W&B with a standardised run name:
`visual-jenga_coco_<model>_n<N>_steps<S>_<date>`

### Run across all datasets with W&B logging

```bash
uv run python run_all_datasets.py \
    --datasets coco nyu clutteredparse \
    --data-root /path/to/datasets \
    --output-root results/datasets \
    --remover lama \
    --n 16 --steps 10 \
    --wandb-project visual-jenga \
    --resume      # skip scenes that already have done.txt
```

### Browse results in the dashboard

```bash
bash open_dashboard.sh   # opens Streamlit on port 8510
```

Then visit `http://localhost:8510` (or `http://<server-ip>:8510` for remote access).

### Download evaluation datasets

```bash
uv run python download_dataset.py --output data/datasets/
```

Downloads `konpat/visual-jenga-datasets` from HuggingFace (~104 MB):
- **NYU** — 668 pairwise comparisons from NYU Depth V2
- **ClutteredParse** — 40 challenging pairwise scenes (HardParse in the paper)
- **COCO** — 200 pairwise scenes
- **FullSceneDecomposition** — 56 complete sequential decompositions

---

## Models Used

| Model | Role | HuggingFace ID |
|---|---|---|
| Molmo-7B-D | Object detection (pointing) | `allenai/Molmo-7B-D-0924` |
| SAM2-Hiera-Large | Segmentation from points | `facebook/sam2-hiera-large` |
| SD1.5 Inpainting | Counterfactual inpainting (diversity scoring) | `runwayml/stable-diffusion-inpainting` |
| LaMa | Final object removal | `big-lama` (auto-downloaded, ~196 MB) |
| CLIP ViT-L/14 | Image similarity | `openai/clip-vit-large-patch14` (via open_clip) |
| DINOv2-Base | Image similarity | `facebook/dinov2-base` |

**Why two inpainting models?**
SD1.5 is used for scoring only — its diversity across N samples is the signal. LaMa is used for the final removal step because it reconstructs clean background without hallucinating replacement objects (which would cause Molmo to re-detect the same location in the next step).

---

## Key Implementation Notes

- **Mask cache**: SAM2 results are cached by detection point. Unchanged objects are not re-segmented across steps — only new points or points near the just-removed object need re-segmentation.
- **Removed-region guard**: A union of all removed masks is accumulated across steps. Detection points falling inside already-removed pixels are silently skipped, preventing infinite re-detection loops.
- **Max steps**: Defaults to 10. Override with `--steps N` (or `--steps 0` for unlimited, though the removed-region guard still applies).
- Crops are square (max of H, W of tight bounding box), resized to 224×224, zero-valued outside the mask.
- Cosine similarity normalised to [0, 1] as `(cos_sim + 1) / 2`.
- `area_fraction` = mask pixels / bounding-box pixels — upweights small objects.
- Mask is morphologically dilated by 10 px before inpainting to avoid edge artefacts.
- Default N=16; paper shows diminishing returns beyond N=8, convergence by N=16.

---

## Inpainting Prompts (from paper Appendix D)

**Positive:** `Full HD, 4K, high quality, high resolution, photorealistic`

**Negative:** `bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality`

---

## Reference

```bibtex
@article{bhattad2025visualjenga,
  title   = {Visual Jenga: Discovering Object Dependencies via Counterfactual Inpainting},
  author  = {Bhattad, Anand and Preechakul, Konpat and Efros, Alexei A.},
  journal = {arXiv preprint arXiv:2503.21770},
  year    = {2025}
}
```
