#!/bin/bash
MODEL="lama"

if [ "$MODEL" == "lama" ]; then
    output_dir="/mnt/sda/edward/data_visualjenga/results/coco_lama"
elif [ "$MODEL" == "sd" ]; then
    output_dir="/mnt/sda/edward/data_visualjenga/results/coco_sd"
fi

uv run python run_all_datasets.py \
    --datasets coco \
    --data-root /mnt/sda/edward/data_visualjenga \
    --output-root $output_dir \
    --remover $MODEL \
    --device cuda:2 \
    --resume
