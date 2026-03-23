#!/usr/bin/env python3
"""
Download the Visual Jenga evaluation datasets from HuggingFace.

Dataset: konpat/visual-jenga-datasets
Contains:
  - NYU          : 668 pairwise scenes (image + mask A + mask B + scene graph)
  - ClutteredParse: 40 challenging pairwise scenes (HardParse in the paper)
  - COCO         : 200 pairwise scenes
  - FullSceneDecomposition: 56 complete scene decomposition sequences

Downloads raw files via huggingface_hub and saves to data/datasets/.
"""

import os
import sys
import argparse
from pathlib import Path


HF_DATASET_ID = "konpat/visual-jenga-datasets"
DEFAULT_OUT = Path(__file__).parent / "data" / "datasets"


def download_raw(output_dir: Path):
    """Download the entire dataset repo as raw files (preserves folder structure)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_DATASET_ID} → {output_dir} ...")
    local_dir = snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print(f"Download complete: {local_dir}")
    return Path(local_dir)


def summarise(dataset_dir: Path):
    """Print a summary of downloaded files."""
    print("\n--- Dataset summary ---")
    for subdir in sorted(dataset_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            files = list(subdir.rglob("*"))
            images = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".avif"}]
            print(f"  {subdir.name}/  — {len(images)} image files")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download Visual Jenga datasets.")
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    out = Path(args.output)
    dataset_dir = download_raw(out)
    summarise(dataset_dir)
    print(f"All done. Data saved to: {dataset_dir}")


if __name__ == "__main__":
    main()
