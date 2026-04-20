#!/bin/bash
# ── Hyperparameters ────────────────────────────────────────────────────────
MODEL="lama"          # lama | sd
N_SAMPLES=3           # inpainting samples per object for diversity scoring
MAX_STEPS=10          # max removals per image
DEVICE="cuda:2"

# ── W&B ───────────────────────────────────────────────────────────────────
WANDB_PROJECT="visual-jenga"
DATE=$(date +%Y-%m-%d)
RUN_NAME="${WANDB_PROJECT}_coco_${MODEL}_n${N_SAMPLES}_steps${MAX_STEPS}_${DATE}"

# ── Derived paths ──────────────────────────────────────────────────────────
OUTPUT_ROOT="/mnt/sda/edward/data_visualjenga/results/coco_${MODEL}"

# ── Run ───────────────────────────────────────────────────────────────────
uv run python run_all_datasets.py \
    --datasets coco \
    --data-root /mnt/sda/edward/data_visualjenga \
    --output-root "$OUTPUT_ROOT" \
    --remover "$MODEL" \
    --n "$N_SAMPLES" \
    --steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$RUN_NAME" \
    --wandb-tags coco "$MODEL" \
    --resume
