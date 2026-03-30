"""
Visual Jenga Dashboard — browse iterative object-removal results.

Usage:
    uv run streamlit run dashboard.py
    uv run streamlit run dashboard.py -- --results results/datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# CLI arg: allow overriding the results root
# ---------------------------------------------------------------------------

def _parse_results_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="results/datasets")
    args, _ = parser.parse_known_args()
    return Path(args.results)


RESULTS_ROOT = _parse_results_root()

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def discover_scenes(results_root: Path) -> dict[str, list[str]]:
    """Return {dataset_name: [scene_id, ...]} sorted."""
    out: dict[str, list[str]] = {}
    if not results_root.exists():
        return out
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        scenes = sorted(
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and (d / "step_00_original.png").exists()
        )
        if scenes:
            out[dataset_dir.name] = scenes
    return out


def get_steps(scene_dir: Path) -> list[int]:
    """Return sorted list of completed step numbers (1-based)."""
    steps = []
    for d in scene_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                n = int(d.name.split("_")[1])
                if n > 0:
                    steps.append(n)
            except (IndexError, ValueError):
                pass
    return sorted(steps)


def load_scores(step_dir: Path) -> list[dict]:
    p = step_dir / "scores.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_detections(step_dir: Path) -> list[dict]:
    p = step_dir / "detect.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def get_mask_viz_paths(step_dir: Path) -> list[Path]:
    return sorted(step_dir.glob("mask_*_viz.png"))


def get_removed_path(step_dir: Path) -> Path | None:
    paths = list(step_dir.glob("removed_*.png"))
    return paths[0] if paths else None


def load_img(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    return Image.open(path)


def is_done(scene_dir: Path) -> bool:
    return (scene_dir / "done.txt").exists()


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def show_overview(scene_dir: Path, steps: list[int]):
    """Show the full removal sequence as a filmstrip."""
    st.subheader("Removal sequence")

    # Collect images: original + removed_* from each step
    images: list[tuple[str, Image.Image]] = []

    orig = load_img(scene_dir / "step_00_original.png")
    if orig:
        images.append(("Original", orig))

    for step_n in steps:
        step_dir = scene_dir / f"step_{step_n:02d}"
        removed = get_removed_path(step_dir)
        if removed:
            images.append((f"After step {step_n}", load_img(removed)))

    if not images:
        st.info("No images found.")
        return

    # Display in rows of 4
    cols_per_row = 4
    for row_start in range(0, len(images), cols_per_row):
        row = images[row_start : row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (caption, img) in zip(cols, row):
            col.image(img, caption=caption, use_container_width=True)


def show_step_detail(scene_dir: Path, step_n: int):
    """Show detailed view of a single step."""
    step_dir = scene_dir / f"step_{step_n:02d}"
    if not step_dir.exists():
        st.warning(f"Step directory not found: {step_dir}")
        return

    # ── Top row: main phase images ───────────────────────────────────────
    orig_img = load_img(step_dir / "original.png")
    detect_img = load_img(step_dir / "detect_viz.png")
    pre_img = load_img(step_dir / "pre_removal.png")
    removed_img = load_img(get_removed_path(step_dir))

    phase_images = [
        ("Input image", orig_img),
        ("Detections", detect_img),
        ("Chosen object", pre_img),
        ("After removal", removed_img),
    ]
    phase_images = [(cap, img) for cap, img in phase_images if img is not None]

    cols = st.columns(len(phase_images))
    for col, (caption, img) in zip(cols, phase_images):
        col.image(img, caption=caption, use_container_width=True)

    # ── Scores bar chart ─────────────────────────────────────────────────
    scores = load_scores(step_dir)
    if scores:
        st.subheader("Diversity scores")
        sorted_scores = sorted(scores, key=lambda s: s["score"], reverse=True)
        labels = [f"obj {s['obj_idx']}" for s in sorted_scores]
        values = [s["score"] for s in sorted_scores]
        colors = ["#FF4B4B" if i == 0 else "#4B8BFF" for i in range(len(values))]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="Diversity score",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red bar = object selected for removal (highest score)")

    # ── Detections table ─────────────────────────────────────────────────
    detections = load_detections(step_dir)
    if detections:
        with st.expander(f"Detections ({len(detections)} objects)"):
            rows = []
            for i, d in enumerate(detections):
                score_val = next(
                    (s["score"] for s in scores if s["obj_idx"] == i), None
                )
                rows.append({
                    "idx": i,
                    "x_frac": round(d["x_frac"], 3),
                    "y_frac": round(d["y_frac"], 3),
                    "score": round(score_val, 4) if score_val is not None else "—",
                })
            st.table(rows)

    # ── Mask gallery ─────────────────────────────────────────────────────
    mask_paths = get_mask_viz_paths(step_dir)
    if mask_paths:
        with st.expander(f"Segmentation masks ({len(mask_paths)})"):
            cols_per_row = 4
            for row_start in range(0, len(mask_paths), cols_per_row):
                row = mask_paths[row_start : row_start + cols_per_row]
                cols = st.columns(len(row))
                for col, p in zip(cols, row):
                    # Extract obj index from filename
                    parts = p.stem.split("_")
                    obj_label = f"obj {parts[1]}" if len(parts) > 1 else p.stem
                    col.image(load_img(p), caption=obj_label, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Visual Jenga",
    page_icon="🧱",
    layout="wide",
)

st.title("Visual Jenga — Results Browser")

scenes_by_dataset = discover_scenes(RESULTS_ROOT)

if not scenes_by_dataset:
    st.error(f"No results found under `{RESULTS_ROOT}`. Run the pipeline first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Navigation")
    st.caption(f"Results root: `{RESULTS_ROOT}`")

    dataset = st.selectbox("Dataset", list(scenes_by_dataset.keys()))
    scenes = scenes_by_dataset[dataset]

    scene_id = st.selectbox("Scene", scenes)
    scene_dir = RESULTS_ROOT / dataset / scene_id

    steps = get_steps(scene_dir)
    done = is_done(scene_dir)

    st.markdown("---")
    st.markdown(f"**Steps completed:** {len(steps)}")
    st.markdown(f"**Status:** {'✅ done' if done else '🔄 in progress'}")

    st.markdown("---")
    view = st.radio("View", ["Overview", "Step detail"])

    if view == "Step detail":
        if steps:
            step_n = st.select_slider(
                "Step",
                options=steps,
                value=steps[0],
            )
        else:
            step_n = None

# ── Main area ─────────────────────────────────────────────────────────────

if view == "Overview":
    show_overview(scene_dir, steps)
else:
    if not steps:
        st.info("No steps found for this scene yet.")
    elif step_n is not None:
        st.subheader(f"Step {step_n} — {dataset} / {scene_id}")
        show_step_detail(scene_dir, step_n)
