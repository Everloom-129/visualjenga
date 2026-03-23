#!/usr/bin/env python3
"""
Visual Jenga — CLI entry point.

Usage:
    python run_jenga.py --image path/to/image.jpg --output results/ [--n 16] [--steps 10] [--device cuda:0]

Outputs step_00_original.png, step_01_pre_removal.png, step_01_removed_<label>.png, …
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

from src.pipeline import VisualJengaPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual Jenga: sequentially remove objects in dependency order."
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to the input image (JPEG/PNG).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Directory where output frames will be saved.",
    )
    parser.add_argument(
        "--n", type=int, default=16,
        help="Number of inpainting samples per object for diversity scoring (default: 16).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Maximum number of objects to remove (default: until empty).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help='CUDA device, e.g. "cuda:0". Auto-detected if not specified.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    print(f"Input image: {img_path}  ({image.size[0]}×{image.size[1]})")
    print(f"Output dir:  {args.output}")
    print(f"n_samples:   {args.n}")
    print(f"max_steps:   {args.steps or 'unlimited'}")
    print(f"device:      {args.device or 'auto'}")
    print()

    pipeline = VisualJengaPipeline(
        device=args.device,
        n_samples=args.n,
        max_steps=args.steps,
        verbose=True,
    )

    frames = pipeline.run(image, output_dir=args.output)
    print(f"\n{len(frames)} frames saved to {args.output}")


if __name__ == "__main__":
    main()
