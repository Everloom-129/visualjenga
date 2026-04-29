---
title: Visual Jenga Reproduction
description: A full reproduction of the Visual Jenga paper — iterative object removal via counterfactual inpainting and diversity scoring.
sidebar_label: Reproduction
slug: /reproduction
---

# Visual Jenga Reproduction

**Author:** [Jie Wang](https://everloom-129.github.io/)

## Introduction

**Visual Jenga** is a method for discovering the dependency order of objects in a scene. Given an image, the pipeline iteratively identifies and removes the "most removable" object — the one whose absence can be filled by many different plausible backgrounds — until the scene is empty. The resulting removal sequence reveals which objects are structural anchors and which are freely replaceable foreground elements.

This page documents a full reproduction of the method described in the Visual Jenga paper ([arXiv:2503.21770](https://arxiv.org/abs/2503.21770)), implemented end-to-end with open-source models.

### Diversity Score

The core signal is the **diversity score** (Equation 2 from the paper):

$$
\text{diversity} = 1 - \frac{\overline{\text{CLIP\_sim}} \times \overline{\text{DINO\_sim}}}{\text{area\_fraction}}
$$

Where:
- $\overline{\text{CLIP\_sim}}$ — mean CLIP cosine similarity between the original object crop and each of the $N$ inpainted samples, mapped to $[0, 1]$ via $(\cos + 1) / 2$
- $\overline{\text{DINO\_sim}}$ — mean DINOv2 cosine similarity between the original crop and each inpainted sample, also in $[0, 1]$
- $\text{area\_fraction}$ — fraction of the tight square bounding box covered by the object mask; normalises for object size

A **high diversity score** means the slot can be filled by many different things → the object is not structurally required → **remove it first**.
A **low score** means all inpaintings look similar to the original → a load-bearing object → **remove it last**.

:::note
The area fraction denominator upweights small objects. Without it, tiny objects would score artificially low simply because their crops vary less in absolute pixel space than large ones.
:::

---

## Pipeline Architecture

The pipeline has two strictly alternating phases per removal step, designed to stay within the VRAM budget of a single 24 GB A10 GPU.

### Model Table

| Module | Class | Model | Role |
|--------|-------|-------|------|
| `detect.py` | `MolmoDetector` | `allenai/Molmo-7B-D-0924` | Prompts with *"Point to every distinct object."*; returns `(label, x_frac, y_frac)` points |
| `segment.py` | `SAM2Segmenter` | `facebook/sam2-hiera-large` | Segments one object per detection point; returns binary foreground mask |
| `inpaint.py` | `SDInpainter` | `runwayml/stable-diffusion-inpainting` | Generates $N = 16$ diverse inpainted samples per object for diversity scoring |
| `similarity.py` | `SimilarityModel` | CLIP ViT-L/14 + `facebook/dinov2-base` | Pairwise cosine similarity between original and inpainted crops |
| `diversity.py` | `diversity_score()` | — (pure NumPy) | Computes Eq. 2 using `SimilarityModel` outputs |
| `inpaint.py` | `LaMaRemover` | `big-lama` | Full-resolution background reconstruction for the final removal frame |
| `pipeline.py` | `VisualJengaPipeline` | — | Orchestrates all of the above; manages load/unload lifecycle |

### Phase A — Detect and Segment

1. **Detect (Molmo-7B):** The 7B multimodal model receives the current image with the fixed prompt `"Point to every distinct object."` and outputs XML-like `<point x="..." y="...">label</point>` tags. Coordinates are in the range $[0, 100]$ and are normalised to $[0, 1]$. A spatial NMS pass (minimum distance 0.05 in fractional coords) removes near-duplicate detections.

2. **Segment (SAM2):** Each surviving detection point is passed to SAM2 as a foreground prompt, producing a binary mask. Duplicate masks (IoU > 0.85) are deduplicated, keeping the larger one. Results are cached by fractional coordinate so that unchanged objects are not re-segmented on subsequent steps.

Molmo is fully unloaded and GPU cache flushed before SAM2 loads; SAM2 is then unloaded before Phase B begins.

### Phase B — Score and Remove

3. **Score (SD 1.5 + CLIP + DINOv2):** For each surviving object, SD 1.5 inpaints $N = 16$ samples into the masked region (512 × 512, seeds 0…15). CLIP ViT-L/14 and DINOv2-base compare each inpainted crop to the original crop. The diversity score (Eq. 2) is computed and written to `scores.json`.

4. **Remove (LaMa):** The object with the highest diversity score is selected. SD + CLIP + DINOv2 are unloaded, LaMa is loaded, and it reconstructs the background at full resolution without hallucinating replacement foreground objects. The result is the input to the next step.

:::tip
Always use `--remover lama` (the default). SD1.5 removal hallucinates replacement objects — a removed person becomes a painted person — causing Molmo to re-detect the same region on the next step and producing very long or non-terminating loops.
:::

---

## Step-by-Step Walkthrough

The following traces one complete rollout on COCO image 000 using the LaMa remover (`--remover lama`). Each step shows the four key outputs in sequence: detection, segmentation, scoring, and removal.

### Original Image

![Original input image](../../static/img/jenga/coco_lama/000/step_00_original.png)

*`step_00_original.png` — the raw input image before any removal.*

---

### Step 1

**Detect:** Molmo scans the full scene and assigns a coloured dot to every distinct object it identifies.

![Step 1 — detection dots](../../static/img/jenga/coco_lama/000/step_01/detect_viz.png)

*`step_01/detect_viz.png` — each coloured dot corresponds to one `(label, x_frac, y_frac)` entry in `detect.json`.*

**Segment:** SAM2 generates a binary mask for each detection point. The mask is saved as a binary PNG (`mask_OO_<label>.png`) and as a semi-transparent red overlay for inspection.

![Step 1 — mask overlay](../../static/img/jenga/coco_lama/000/step_01/mask_00_distinct_object__viz.png)

*`step_01/mask_00_distinct_object__viz.png` — red-tint overlay of the segmented region.*

**Score:** SD 1.5 generates 16 inpainting samples per object; CLIP and DINOv2 compare each to the original crop; the diversity score (Eq. 2) is recorded in `scores.json`.

**Remove:** The object with the highest diversity score is annotated with a yellow dot, then cleanly removed by LaMa.

![Step 1 — pre-removal annotation](../../static/img/jenga/coco_lama/000/step_01/pre_removal.png)

*`step_01/pre_removal.png` — yellow dot and score label mark the chosen object before removal.*

![Step 1 — after removal](../../static/img/jenga/coco_lama/000/step_01/removed_distinct_object_.png)

*`step_01/removed_distinct_object_.png` — LaMa fills the masked region with plausible background texture.*

---

### Step 2

The image produced by step 1 is passed back to Molmo for re-detection. Detection points inside the `removed_union` mask are skipped before segmentation.

![Step 2 — image entering step](../../static/img/jenga/coco_lama/000/step_02/original.png)

*`step_02/original.png` — the current image entering step 2.*

![Step 2 — detection dots](../../static/img/jenga/coco_lama/000/step_02/detect_viz.png)

*`step_02/detect_viz.png` — Molmo re-detects objects on the modified scene. Points in already-removed regions are filtered out by `removed_union`.*

![Step 2 — pre-removal annotation](../../static/img/jenga/coco_lama/000/step_02/pre_removal.png)

*`step_02/pre_removal.png` — the next highest-scoring object is selected for removal.*

![Step 2 — after removal](../../static/img/jenga/coco_lama/000/step_02/removed_distinct_object_.png)

*`step_02/removed_distinct_object_.png` — background reconstruction after the second removal.*

---

### Step 3

![Step 3 — image entering step](../../static/img/jenga/coco_lama/000/step_03/original.png)

*`step_03/original.png` — the scene after two objects have been removed.*

![Step 3 — detection dots](../../static/img/jenga/coco_lama/000/step_03/detect_viz.png)

*`step_03/detect_viz.png` — Molmo detects the remaining objects. The pipeline continues until no objects are found or `max_steps` is reached.*

![Step 3 — pre-removal annotation](../../static/img/jenga/coco_lama/000/step_03/pre_removal.png)

*`step_03/pre_removal.png` — third object selected for removal.*

![Step 3 — after removal](../../static/img/jenga/coco_lama/000/step_03/removed_distinct_object_.png)

*`step_03/removed_distinct_object_.png` — the scene progressively simplified toward an empty background.*

---

### Output Directory Layout

Each run produces the following structure under the specified `--output` directory:

```
output_dir/
  step_00_original.png          ← raw input image
  step_NN/
    original.png                ← image entering this step
    detect.json                 ← raw Molmo output: [{label, x_frac, y_frac}, ...]
    detect_viz.png              ← coloured detection dots overlay
    mask_OO_<label>.png         ← binary mask (255 = object, 0 = background)
    mask_OO_<label>_viz.png     ← red-tint overlay on current image
    scores.json                 ← diversity score per detected object
    inpaint_OO_SS_<label>.png   ← (optional) SD samples used for scoring
    pre_removal.png             ← chosen object with yellow dot + score
    removed_<label>.png         ← image after removal (input to step N+1)
```

---

## LaMa vs SD1.5 Comparison

The pipeline supports two removal backends, selectable via `--remover lama` (default) or `--remover sd`.

### Same Scene, Two Backends

| LaMa (default) | SD1.5 |
|----------------|-------|
| ![LaMa step 1 removal](../../static/img/jenga/coco_lama/000/step_01/removed_distinct_object_.png) | ![SD step 1 removal](../../static/img/jenga/coco_sd/000/step_01/removed_distinct_object_.png) |
| Clean background reconstruction | May hallucinate a replacement foreground object |

### Why LaMa Is Preferred

**SD1.5 hallucinates.** Stable Diffusion is a generative model: when asked to fill the region where a person stood, it will often paint in *another* person, a piece of furniture, or another plausible-but-spurious foreground object. On the next step, Molmo detects this hallucinated object in the same region. The pipeline attempts to remove it, SD replaces it with yet another hallucination, and the loop continues — exhausting `max_steps` without making real progress through the scene.

**LaMa reconstructs.** LaMa (Large Mask inpainting) is specifically trained to reconstruct natural backgrounds. It operates at full resolution without crop-resize artefacts and consistently fills removed regions with texturally plausible backgrounds — walls, floors, sky — rather than new foreground objects. This allows the `removed_union` guard to function correctly.

:::warning
Use `--remover sd` only when explicitly reproducing the original paper's behaviour for a controlled comparison. All production runs and quantitative evaluations should use `--remover lama`.
:::

---

## GPU Memory Management

### The Problem

Molmo-7B in bfloat16 requires approximately **14 GB** of VRAM. SAM2 + SD 1.5 + CLIP ViT-L/14 + DINOv2-base together require approximately **6 GB**. Both groups cannot be resident simultaneously on a 24 GB A10 GPU.

### Two-Phase Load / Unload Strategy

Each pipeline step is split into two strictly separated phases. No model reference is held across a phase boundary — a stale Python reference would prevent the garbage collector from freeing the underlying tensors even after `unload()` is called.

**Phase A — Detect and Segment (≤ 14 GB peak)**

```
Load Molmo-7B
  → detect all objects
  → unload + gc.collect() + torch.cuda.empty_cache()

Load SAM2
  → segment each detection point
  → unload + gc.collect() + torch.cuda.empty_cache()
```

**Phase B — Score and Remove (≤ 6 GB peak, LaMa remover)**

```
Load SD 1.5 + CLIP ViT-L/14 + DINOv2
  → score all objects (diversity_score())
  → unload all three + gc.collect() + torch.cuda.empty_cache()

Load LaMa
  → remove the highest-scoring object
  → unload + gc.collect() + torch.cuda.empty_cache()
```

Every model class exposes an `unload()` method that sets all internal PyTorch references to `None`. The helper `_free_gpu()` in `pipeline.py` then calls `gc.collect()` followed by `torch.cuda.empty_cache()` to return VRAM to the CUDA allocator before the next model loads.

:::note
Do not hold model references across phase boundaries. Even a loop variable left in scope can block garbage collection and silently leak VRAM.
:::

### Loop Guards

Two mechanisms prevent runaway loops:

1. **`removed_union` mask** — a boolean array of shape $(H, W)$ accumulating every pixel removed so far. Before segmentation, any Molmo detection point whose pixel coordinate falls inside `removed_union` is silently dropped. This prevents the pipeline from re-detecting a successfully removed region regardless of what the background fill looks like.

2. **`max_steps` cap** — a hard upper bound on the number of removals per image (default: 10; override with `--steps N`). The main loop exits unconditionally when this limit is reached.

### Mask Cache

To avoid redundant SAM2 calls when Molmo re-detects the same object across consecutive steps, the pipeline maintains `_mask_cache: dict[(x_frac, y_frac) → bool mask]`:

- **Hit radius:** 0.02 (fractional image coordinates) — if a new detection point falls within 2% of a cached key, the stored mask is reused without calling SAM2.
- **Invalidation radius:** 0.10 — after each removal, all cache entries within 10% of the removed object's point are evicted, forcing neighbours to re-segment since the surrounding scene has changed.

---

## Quantitative Results

Results on standard benchmarks. Full experiments in progress.

| Dataset | Remover | Avg Steps | Notes |
|---------|---------|-----------|-------|
| COCO (200 images) | LaMa | TBD | — |
| COCO (200 images) | SD1.5 | TBD | Loop artefacts inflate step count |
| NYU Depth | LaMa | TBD | — |
| ClutteredParse | LaMa | TBD | — |

:::note
Results will be updated as experiments complete. Use `--resume` with `run_all_datasets.py` to skip images that already have a `done.txt` marker.
:::

---

## How to Run

**Install dependencies (first time):**

```bash
uv sync
```

**Run on a single image:**

```bash
uv run python run_jenga.py \
    --image data/datasets/coco/000/img.jpeg \
    --output results/ \
    --remover lama \
    --n 16 \
    --steps 10
```

**Run across all datasets with Weights & Biases logging:**

```bash
uv run python run_all_datasets.py \
    --datasets coco nyu clutteredparse \
    --remover lama \
    --wandb-project visual-jenga \
    --resume
```

**Open the Streamlit result browser:**

```bash
bash open_dashboard.sh   # → http://localhost:8510
```

**Run the test suite:**

```bash
bash run_tests.sh             # unit tests only (no GPU required)
bash run_tests.sh --smoke     # unit + quick GPU sanity check (1 forward pass per model)
bash run_tests.sh --all       # full suite including slow GPU tests
```

:::tip
Run `bash run_tests.sh` before submitting results to verify the diversity formula, crop logic, and similarity utilities are all behaving correctly.
:::

---

## Citation and Acknowledgements

This page documents a reproduction of the Visual Jenga method. Please cite the original paper if you use or build on this work:

```bibtex
@article{bhattad2025visualjenga,
  title   = {Visual Jenga: Discovering Object Dependencies via Counterfactual Inpainting},
  author  = {Bhattad, Anand and Preechakul, Konpat and Efros, Alexei A.},
  journal = {arXiv preprint arXiv:2503.21770},
  year    = {2025},
}
```

Reproduction implemented by **[Jie Wang](https://everloom-129.github.io/)**.

Open-source models used in this reproduction:

- [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) — Allen Institute for AI
- [SAM2 Hiera Large](https://huggingface.co/facebook/sam2-hiera-large) — Meta AI
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) — RunwayML
- [DINOv2 Base](https://huggingface.co/facebook/dinov2-base) — Meta AI
- [CLIP ViT-L/14](https://github.com/openai/CLIP) — OpenAI
- [LaMa](https://github.com/advimman/lama) — Samsung Research
