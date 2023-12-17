#!/bin/bash

codec_name="tmc3"
codec_type="tmc3"
ref_dir_root="/mnt/data/datasets/modelnet/by_n_ply/1024"
in_dir_root="/mnt/data/datasets/modelnet/by_n_ply"
out_dir_root="/mnt/data/datasets/modelnet/by_n_scale_ply_${codec_name}"
# TODO supply format string instead of hardcoding...

NUM_POINTSES=(8 16 32 64 128 256 512 1024)
SCALES=(8 16 32 64 128 256)

for num_points in "${NUM_POINTSES[@]}"; do
  num_points_str="$(printf '%04d' "$num_points")"

  for scale in "${SCALES[@]}"; do
    scale_str="$(printf '%04d' "$scale")"

    python scripts/generate_input_codec_dataset.py \
      --codec="${codec_type}" \
      --out_results_tsv="input_codec_dataset_${codec_name}.tsv" \
      --ref_dir="${ref_dir_root}" \
      --in_dir="${in_dir_root}/${num_points_str}" \
      --out_dir="${out_dir_root}/${num_points_str}/${scale_str}" \
      --scale="${scale}"
  done
done
