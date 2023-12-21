#!/bin/bash

codec_name="tmc3"
in_dir_root="/mnt/data/datasets/modelnet/by_n_scale_ply_${codec_name}"
runs_root="$HOME/data/runs/pc-mordor"

NUM_POINTSES=(8 16 32 64 128 256 512 1024)
SCALES=(8 16 32 64 128 256)

# PointNet varying num_points models.
RUN_HASHES=(
  "f84aa21a1ce3491d800733a6"  # 8
  "552499a134ac41cebc50e967"  # 16
  "d19f6ea297bb42b1902c4678"  # 32
  "010feb1990e44251ad080a35"  # 64
  "bad325156b3e4137bc7752d9"  # 128
  "e2f872d550c548f2b687c88a"  # 256
  "5d0ef9a2a12b4752b11cd61b"  # 512
  "b8d279f32b3f43d89d386b11"  # 1024
  # "e3eeddad2cc64348b4f03c89"  # 2048
)

# PointNet++ varying num_points models.
RUN_HASHES=(
  "9f6eacf568bc4ba382dc0dc0"  # 16
  "5fb74416eac946bfbb4a2b0e"  # 32
  "2127e9d1fdbe4b56ba4810d6"  # 64
  "65a60fd323e747f99524ce2c"  # 128
  "8c204568630041488dc73c1c"  # 256
  "12cf93616a094c349a99e167"  # 512
  "792d24083f5f4cef893e8d2a"  # 1024
  # "3a022c5df91745c48c219aae"  # 2048
  # "bbd9be0a3b584813b01b87d7"  # 4096
  # "9970ba7cbc994da883e66609"  # 8192
)


for ((i = 0; i < "${#NUM_POINTSES[@]}"; i += 1)); do
  num_points="${NUM_POINTSES[$i]}"
  num_points_str="$(printf '%04d' "$num_points")"
  run_hash="${RUN_HASHES[$i]}"

  for scale in "${SCALES[@]}"; do
    scale_str="$(printf '%04d' "$scale")"

    python scripts/eval_input_codec.py \
      --config-path="${runs_root}/${run_hash}/configs" \
      --config-name=config \
      --codec="${codec_name}" \
      --in_results_tsv="input_codec_dataset_${codec_name}.tsv" \
      --in_dir="${in_dir_root}/${num_points_str}/${scale_str}" \
      --scale="${scale}"
  done
done
