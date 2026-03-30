#!/usr/bin/env python3
"""
Run VisualJengaPipeline across all (or selected) datasets and log results to W&B.

Usage:
    uv run python run_all_datasets.py [options]

Examples:
    # Run everything with defaults
    uv run python run_all_datasets.py

    # Run only COCO and ClutteredParse, limit to 5 objects per image
    uv run python run_all_datasets.py --datasets coco clutteredparse --steps 5

    # Resume a previous run (skips images that already have outputs)
    uv run python run_all_datasets.py --resume

    # Dry run — discover scenes and log to W&B without running the pipeline
    uv run python run_all_datasets.py --dry-run

Datasets (in data/datasets/):
    coco              200 pairwise scenes (img.jpeg + A.png + B.png)
    nyu               668 pairwise scenes (img.jpg + A.png + B.png + scene_graph)
    clutteredparse     40 challenging pairwise scenes
    full_scene_decom   56 flat images (no A/B masks)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
_ALL_DATASETS = ["coco", "nyu", "clutteredparse", "full_scene_decom"]


def find_scenes(dataset_root: Path, dataset: str) -> list[dict]:
    """Return sorted list of scene dicts:
        scene_id    str
        image_path  Path
        mask_a      Path | None   (ground-truth "before" mask)
        mask_b      Path | None   (ground-truth "after" mask)
    """
    root = dataset_root / dataset
    if not root.exists():
        print(f"[warn] Dataset directory not found: {root} — skipping")
        return []

    scenes = []

    if dataset == "full_scene_decom":
        # Flat directory — each image file is its own scene
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith("."):
                scenes.append({"scene_id": f.stem, "image_path": f, "mask_a": None, "mask_b": None})
    else:
        # One sub-directory per scene
        for scene_dir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
            img_path = _find_image(scene_dir)
            if img_path is None:
                continue
            mask_a = scene_dir / "A.png"
            mask_b = scene_dir / "B.png"
            scenes.append({
                "scene_id": scene_dir.name,
                "image_path": img_path,
                "mask_a": mask_a if mask_a.exists() else None,
                "mask_b": mask_b if mask_b.exists() else None,
            })

    return scenes


def _find_image(scene_dir: Path) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        p = scene_dir / f"img{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# W&B step callback
# ---------------------------------------------------------------------------

def make_step_callback(
    wandb_run,
    dataset: str,
    scene_id: str,
    image_idx: int,
):
    """Return a callback compatible with VisualJengaPipeline.run(on_step_complete=...)."""

    def callback(
        step: int,
        detected,           # list[DetectedObject]
        scored,             # list[ScoredObject], sorted descending by score
        step_dir: str,
    ):
        import wandb

        step_path = Path(step_dir)
        log: dict = {
            "step/image_idx":        image_idx,
            "step/dataset":          dataset,
            "step/scene_id":         scene_id,
            "step/step_num":         step,
            "step/n_detected":       len(detected),
            "step/n_scored":         len(scored),
        }

        if scored:
            removed = scored[0]
            log["step/removed_label"]        = removed.detected.label
            log["step/max_diversity_score"]  = removed.score
            log["step/min_diversity_score"]  = scored[-1].score

        # Images already on disk — upload them
        for fname, key in [
            ("detect_viz.png",  "step/detect_viz"),
            ("pre_removal.png", "step/pre_removal"),
        ]:
            p = step_path / fname
            if p.exists():
                log[key] = wandb.Image(str(p))

        # Removal result (removed_<label>.png — glob for it)
        removed_imgs = sorted(step_path.glob("removed_*.png"))
        if removed_imgs:
            log["step/removed_result"] = wandb.Image(str(removed_imgs[0]))

        # Per-object score table
        if scored:
            table = wandb.Table(
                columns=["rank", "label", "diversity_score"],
                data=[
                    [rank + 1, s.detected.label, round(s.score, 5)]
                    for rank, s in enumerate(scored)
                ],
            )
            log["step/scores_table"] = table

        wandb_run.log(log)

    return callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Visual Jenga on all datasets with W&B logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=_ALL_DATASETS,
                   choices=_ALL_DATASETS, metavar="DS",
                   help="Datasets to run. Choices: " + ", ".join(_ALL_DATASETS))
    p.add_argument("--data-root", default="data/datasets",
                   help="Root directory containing dataset sub-folders.")
    p.add_argument("--output-root", default="results/datasets",
                   help="Root directory for pipeline outputs.")
    p.add_argument("--n", type=int, default=16,
                   help="Inpainting samples per object for diversity scoring.")
    p.add_argument("--steps", type=int, default=None,
                   help="Max objects to remove per image (None = until empty).")
    p.add_argument("--device", default=None,
                   help='CUDA device, e.g. "cuda:0". Auto-detected if omitted.')
    p.add_argument("--resume", action="store_true",
                   help="Skip images that already have a done.txt sentinel.")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover scenes and log metadata to W&B without running the pipeline.")
    # W&B
    p.add_argument("--wandb-project", default="visual-jenga",
                   help="W&B project name.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team or username). Uses default if omitted.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run display name. Auto-generated if omitted.")
    p.add_argument("--wandb-tags", nargs="*", default=[],
                   help="Extra W&B tags to attach to the run.")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # ── Discover all scenes ────────────────────────────────────────────────
    all_scenes: list[tuple[str, dict]] = []   # (dataset_name, scene_dict)
    for ds in args.datasets:
        scenes = find_scenes(data_root, ds)
        for s in scenes:
            all_scenes.append((ds, s))
        print(f"  {ds}: {len(scenes)} scenes")
    print(f"Total: {len(all_scenes)} scenes across {len(args.datasets)} dataset(s)\n")

    if not all_scenes:
        print("No scenes found — exiting.")
        sys.exit(0)

    # ── W&B init ──────────────────────────────────────────────────────────
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=["batch"] + args.datasets + args.wandb_tags,
        config={
            "datasets":    args.datasets,
            "n_samples":   args.n,
            "max_steps":   args.steps,
            "device":      args.device or "auto",
            "total_scenes": len(all_scenes),
            "resume":      args.resume,
            "dry_run":     args.dry_run,
        },
    )
    print(f"W&B run: {run.url}\n")

    # Summary table accumulated across all images
    summary_table = wandb.Table(columns=[
        "image_idx", "dataset", "scene_id",
        "steps_taken", "duration_s", "status", "error",
    ])

    # ── Pipeline init (reused across all images) ──────────────────────────
    pipeline = None
    if not args.dry_run:
        from src.pipeline import VisualJengaPipeline
        pipeline = VisualJengaPipeline(
            device=args.device,
            n_samples=args.n,
            max_steps=args.steps,
            verbose=True,
        )

    # ── Main loop ─────────────────────────────────────────────────────────
    n_done = 0
    n_skipped = 0
    n_errors = 0

    for image_idx, (dataset, scene) in enumerate(all_scenes):
        scene_id   = scene["scene_id"]
        image_path = scene["image_path"]
        output_dir = output_root / dataset / scene_id
        done_file  = output_dir / "done.txt"

        print(f"[{image_idx + 1}/{len(all_scenes)}] {dataset}/{scene_id}")

        # Resume: skip completed
        if args.resume and done_file.exists():
            print(f"  → already done, skipping")
            n_skipped += 1
            summary_table.add_data(image_idx, dataset, scene_id, None, None, "skipped", "")
            continue

        # Log scene metadata immediately (visible in W&B even if pipeline crashes)
        scene_log: dict = {
            "scene/image_idx": image_idx,
            "scene/dataset":   dataset,
            "scene/scene_id":  scene_id,
        }
        try:
            scene_log["scene/input"] = wandb.Image(str(image_path))
        except Exception:
            pass
        if scene["mask_a"]:
            try:
                scene_log["scene/mask_a"] = wandb.Image(str(scene["mask_a"]))
            except Exception:
                pass
        if scene["mask_b"]:
            try:
                scene_log["scene/mask_b"] = wandb.Image(str(scene["mask_b"]))
            except Exception:
                pass
        run.log(scene_log)

        if args.dry_run:
            summary_table.add_data(image_idx, dataset, scene_id, 0, 0.0, "dry_run", "")
            n_done += 1
            continue

        # ── Run pipeline ──────────────────────────────────────────────────
        t0 = time.time()
        status = "done"
        error_msg = ""
        steps_taken = 0

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            image = Image.open(image_path).convert("RGB")

            callback = make_step_callback(run, dataset, scene_id, image_idx)

            frames = pipeline.run(
                image=image,
                output_dir=str(output_dir),
                on_step_complete=callback,
            )
            steps_taken = len(frames) - 1  # frames[0] is the original

            # Write sentinel so --resume can skip this scene
            done_file.write_text(f"steps={steps_taken}\n")
            n_done += 1

        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            traceback.print_exc()
            n_errors += 1

        duration = time.time() - t0

        # Per-image summary log
        run.log({
            "image/image_idx":   image_idx,
            "image/dataset":     dataset,
            "image/scene_id":    scene_id,
            "image/steps_taken": steps_taken,
            "image/duration_s":  round(duration, 1),
            "image/status":      status,
        })
        summary_table.add_data(
            image_idx, dataset, scene_id,
            steps_taken, round(duration, 1), status, error_msg,
        )

        print(f"  → {status} | steps={steps_taken} | {duration:.0f}s")

    # ── Final summary ─────────────────────────────────────────────────────
    run.log({
        "summary/total":    len(all_scenes),
        "summary/done":     n_done,
        "summary/skipped":  n_skipped,
        "summary/errors":   n_errors,
        "summary/table":    summary_table,
    })

    print(f"\nFinished: {n_done} done, {n_skipped} skipped, {n_errors} errors")
    print(f"W&B run:  {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
