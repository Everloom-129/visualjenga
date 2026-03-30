# CLAUDE.md

## Project Overview

**Visual Jenga** — discovers object dependencies via counterfactual inpainting. Given an image, iteratively removes the "most removable" object (highest diversity score) until nothing is left, producing a dependency-ordered sequence.

Based on the Visual Jenga paper. Diversity score (Eq. 2):
```
diversity = 1 - (mean_CLIP_sim × mean_DINO_sim) / area_fraction
```
High score = the slot can be filled by many things = remove this object first.

## Package Manager

**uv** exclusively. Always prefix Python commands with `uv run`.

```bash
uv sync                   # install dependencies (first time)
uv run python run_jenga.py --image data/datasets/coco/000/img.jpeg --output results/
```

## Key Commands

```bash
# Run the pipeline on a single image
uv run python run_jenga.py --image <path> --output <dir> [--n 16] [--steps N] [--device cuda:0]

# Convenience script (uses data/datasets/coco/000/img.jpeg → results/)
bash run_jenga.sh

# Run across all datasets with W&B logging
uv run python run_all_datasets.py [--datasets coco nyu clutteredparse] \
    [--output-root results/datasets] [--n 16] [--steps N] \
    [--wandb-project visual-jenga] [--wandb-entity <team>] \
    [--resume]          # skip images that already have done.txt
    [--dry-run]         # discover scenes + log to W&B without running inference

# Tests
bash run_tests.sh            # unit tests only (fast, no GPU)
bash run_tests.sh --smoke    # unit + quick GPU sanity (1 forward pass per model)
bash run_tests.sh --all      # everything including slow GPU tests
bash run_tests.sh --gpu      # GPU tests only

# Or call pytest directly
uv run pytest tests/ -m "not gpu and not slow" -v   # unit only
uv run pytest tests/ -m "gpu and not slow" -v       # smoke GPU
```

## Architecture

### Source modules (`src/`)

| File | Class | Model / Role |
|------|-------|------|
| `detect.py` | `MolmoDetector` | allenai/Molmo-7B-D-0924 — points at every distinct object |
| `segment.py` | `SAM2Segmenter` | facebook/sam2-hiera-large — masks one object per point |
| `inpaint.py` | `SDInpainter` | runwayml/stable-diffusion-inpainting — fills masked regions |
| `similarity.py` | `SimilarityModel` | CLIP ViT-L/14 + facebook/dinov2-base — pairwise similarity |
| `diversity.py` | `diversity_score()` | Pure numpy — computes Eq. 2 using `SimilarityModel` |
| `pipeline.py` | `VisualJengaPipeline` | Orchestrates all of the above |

### Pipeline loop

Each step of the loop has two phases that must be kept separate (GPU memory):

**Phase A** — load, run, unload before Phase B:
1. **Detect** (Molmo): points at every object → `detect.json`, `detect_viz.png`
2. **Segment** (SAM2): masks each point → `mask_OO_<label>.png` + overlay

**Phase B** — load SD + CLIP + DINO together, then unload:
3. **Score** each object: SD inpaints N samples → CLIP/DINO diversity score → `scores.json`
4. **Remove** highest-scoring object → `removed_<label>.png`

### Output directory layout

```
output_dir/
  step_00_original.png
  step_NN/
    original.png              — image entering this step
    detect.json               — raw Molmo output (label, x_frac, y_frac)
    detect_viz.png            — colored detection dots
    mask_OO_<label>.png       — binary mask (255 = object)
    mask_OO_<label>_viz.png   — red-tint overlay on image
    scores.json               — diversity score per object
    inpaint_OO_SS_<label>.png — (optional) SD samples used for scoring
    pre_removal.png           — chosen object annotated with yellow dot
    removed_<label>.png       — image after removal (input to next step)
```

## GPU Memory Management

Molmo-7B alone takes ~14 GB bfloat16. SAM2 + SD + CLIP + DINO together take ~5 GB. Both cannot be loaded simultaneously on a 24 GB GPU (A10).

The pipeline explicitly loads and unloads each model group per step. Every model class has an `unload()` method; after unloading, `_free_gpu()` calls `gc.collect()` + `torch.cuda.empty_cache()`. **Do not hold model references across phases.**

## Tests

Tests live in `tests/`. Two pytest marks gate hardware-dependent tests:

- `@pytest.mark.gpu` — requires CUDA + downloaded models
- `@pytest.mark.slow` — multiple inpaintings / long forward passes

Unit tests (no GPU) cover: Molmo output parsing, SAM2 mask dedup, SD pipeline, CLIP/DINO similarity, crop/bbox logic, diversity formula (with a `MockSimilarityModel`).

Fixtures are defined in `tests/conftest.py`: `small_rgb_image` (64×64), `square_mask`, `thin_mask`, `multi_object_masks`.

## Data

Example datasets live in `data/datasets/`. Download with:
```bash
uv run python download_dataset.py
```
