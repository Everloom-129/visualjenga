#!/bin/bash
# ── Config ─────────────────────────────────────────────────────────────────
RESULTS_ROOT="/mnt/sda/edward/data_visualjenga/results/coco_lama"
PORT=8510

# ── Launch ─────────────────────────────────────────────────────────────────
uv run streamlit run dashboard.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --results "$RESULTS_ROOT"
