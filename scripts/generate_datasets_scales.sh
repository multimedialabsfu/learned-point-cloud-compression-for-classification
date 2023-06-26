#!/bin/bash

OUT_TSV="results/tmc13_file_sizes.tsv"
by_n_dir="by_n_ply"
by_n_scale_dir="by_n_scale_ply"


# TSV header
printf '%s\t%s\t%s\n' "infile" "binfile" "size" >> "${OUT_TSV}"

for N in 8 16 32 64 128 256 512 1024; do
  N_str="$(printf '%04d' "$N")"

  for scale in 1 2 4 8 16 32 64 128 256; do
    scale_str="$(printf '%04d' "$scale")"

    fd . "${by_n_dir}/${N_str}" -e ply | while read -r infile; do
      # infile: by_n/$N/modelnet40/filename.ply
      # binfile: by_n_scale/$N/$scale/modelnet40/filename.ply
      #
      subpath="${infile#"${by_n_dir}/${N_str}/"}"
      binfile="${by_n_scale_dir}/${N_str}/${scale_str}/${subpath%.ply}.bin"
      recfile="$binfile.ply"

      printf "%s\n  => %s\n  => %s\n" "$infile" "$binfile" "$recfile"
      mkdir -p "${binfile%/*}" "${recfile%/*}"

      # Compress.
      tmc3 --mode=0 --uncompressedDataPath="$infile" --compressedStreamPath="$binfile" --inputScale="$scale" >/dev/null

      # Decompress.
      size="$(
        tmc3 --mode=1 --compressedStreamPath="$binfile" --reconstructedDataPath="$recfile" --outputBinaryPly=0 |
        sed -n 's/^positions bitstream size \([0-9]\+\) B$/\1/p'
      )"

      printf '%s\t%s\t%d\n' "$infile" "$binfile" "$size" >> "${OUT_TSV}"
    done
  done
done

