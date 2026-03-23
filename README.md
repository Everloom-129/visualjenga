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
    │
    ▼
[3] Score each object   — For each mask:
    │                       • Inpaint the masked region N=16 times (SD1.5)
    │                       • Crop each result to 224×224 (zero outside mask)
    │                       • Compute CLIP + DINO similarity vs. original crop
    │                       • diversity = 1 − mean(ClipSim)×mean(DinoSim) / area_frac
    ▼
[4] Select & remove     — Highest diversity object → SD1.5 inpainting (final removal)
    │
    ▼
[5] Repeat              — Re-run on updated image until background only
```

**Diversity score (Eq. 2 from paper):**

```
diversity(A) = 1 − [ (1/N Σ ClipSim(c_new_j, c_orig)) × (1/N Σ DinoSim(c_new_j, c_orig)) ] / area_fraction
```

---

## Project Structure

```
visualjenga/
├── run_jenga.py              # CLI entry point
├── download_dataset.py       # Download evaluation datasets from HuggingFace
├── pyproject.toml            # Dependencies (uv)
├── src/
│   ├── detect.py             # Molmo-7B object detection → (x, y) points
│   ├── segment.py            # SAM2 segmentation from point prompts
│   ├── inpaint.py            # SD1.5 inpainting (N samples + final removal)
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

### Run on an image

```bash
uv run python run_jenga.py \
    --image path/to/scene.jpg \
    --output results/ \
    --n 16              # inpainting samples per object (default 16, use 4 for quick test)
    --steps 10          # max objects to remove (default: unlimited)
    --device cuda:0
```

Output frames are saved as:
```
results/
├── step_00_original.png
├── step_01_pre_removal.png        # yellow dot marks the selected object
├── step_01_removed_cat.png        # image after removal
├── step_02_pre_removal.png
├── step_02_removed_laptop.png
└── ...
```

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
| SD1.5 Inpainting | Counterfactual inpainting + removal | `runwayml/stable-diffusion-inpainting` |
| CLIP ViT-L/14 | Image similarity | `openai/clip-vit-large-patch14` (via open_clip) |
| DINOv2-Base | Image similarity | `facebook/dinov2-base` |

---

## Inpainting Prompts (from paper Appendix D)

**Positive:** `Full HD, 4K, high quality, high resolution, photorealistic`

**Negative:** `bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality`

---

## Key Implementation Notes

- Crops are square (max of H, W of tight bounding box), resized to 224×224, zero-valued outside the mask
- Cosine similarity normalized to [0, 1] as `(cos_sim + 1) / 2`
- `area_fraction` = mask pixels / bounding-box pixels — upweights small objects
- Mask is morphologically dilated by 10px before inpainting to avoid edge artifacts
- Default N=16; paper shows diminishing returns beyond N=8, convergence by N=16

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
