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
uv run python run_jenga.py --image <path> --output <dir> \
    [--n 16] [--steps 10] [--remover lama|sd] [--device cuda:0]

# Run full COCO dataset (edit MODEL= at top of script)
bash process_coco.sh

# Run across all datasets with W&B logging
uv run python run_all_datasets.py [--datasets coco nyu clutteredparse] \
    [--data-root /path/to/data] [--output-root results/datasets] \
    [--n 16] [--steps 10] [--remover lama|sd] \
    [--wandb-project visual-jenga] [--wandb-entity <team>] \
    [--resume]          # skip images that already have done.txt
    [--dry-run]         # discover scenes + log to W&B without running inference

# Open Streamlit result browser
bash open_dashboard.sh   # → http://localhost:8510

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
| `inpaint.py` | `SDInpainter` | runwayml/stable-diffusion-inpainting — N diverse samples for scoring |
| `inpaint.py` | `LaMaRemover` | big-lama — clean background removal for the final output frame |
| `similarity.py` | `SimilarityModel` | CLIP ViT-L/14 + facebook/dinov2-base — pairwise similarity |
| `diversity.py` | `diversity_score()` | Pure numpy — computes Eq. 2 using `SimilarityModel` |
| `pipeline.py` | `VisualJengaPipeline` | Orchestrates all of the above |

### Pipeline loop

Each step of the loop has two phases that must be kept separate (GPU memory):

**Phase A** — load, run, unload before Phase B:
1. **Detect** (Molmo): points at every object → `detect.json`, `detect_viz.png`
2. **Segment** (SAM2): masks each point → `mask_OO_<label>.png` + overlay
   - Results cached by detection point; SAM2 only runs for new/changed points
   - Points inside already-removed regions are skipped entirely

**Phase B** — load SD + CLIP + DINO, score, unload, then load LaMa:
3. **Score** each object: SD inpaints N samples → CLIP/DINO diversity score → `scores.json`
4. **Remove** highest-scoring object → LaMa clean removal → `removed_<label>.png`

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

Molmo-7B alone takes ~14 GB bfloat16. SAM2 + SD + CLIP + DINO + LaMa together take ~6 GB. Both cannot be loaded simultaneously on a 24 GB GPU (A10).

The pipeline explicitly loads and unloads each model group per step. Every model class has an `unload()` method; after unloading, `_free_gpu()` calls `gc.collect()` + `torch.cuda.empty_cache()`. **Do not hold model references across phases.**

Phase B load order (lama remover):
1. Load SD + CLIP + DINO → score all objects → unload all three
2. Load LaMa → remove chosen object → unload LaMa

## Remover Backends

`VisualJengaPipeline(remover="lama")` (default) uses LaMa for the final removal step.
`VisualJengaPipeline(remover="sd")` uses SD1.5 for both scoring and removal (original behaviour).

**Always use LaMa unless explicitly comparing against SD.** SD1.5 hallucinate replacement objects, causing Molmo to re-detect the same region in subsequent steps and producing extremely long loops.

## Loop Guards

Two mechanisms prevent infinite loops:

1. **`removed_union`**: Union of all masks removed so far. Detection points falling inside this union are skipped before segmentation. Reset at the start of each `run()` call.
2. **`max_steps`**: Hard cap on removals per image (default: 10). Override with `--steps N`.

## Mask Cache

`pipeline._mask_cache` maps `(x_frac, y_frac) → bool mask`. After each SAM2 call, results are stored. On subsequent steps, if Molmo detects a point within `_CACHE_HIT_RADIUS=0.02` of a cached point, the cached mask is reused (no SAM2 call). After a removal, all entries within `_CACHE_INVALIDATION_RADIUS=0.10` of the removed object's point are evicted so neighbours get re-segmented.

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

Actual experiment data lives at `/mnt/sda/edward/data_visualjenga/`:
```
/mnt/sda/edward/data_visualjenga/
  coco/                    # input images (000–200)
  results/
    coco_lama/             # outputs from LaMa remover run
    coco_sd/               # outputs from SD1.5 remover run
```
