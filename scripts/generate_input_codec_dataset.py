from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def tmc3_compress(in_file, bin_file, **kwargs):
    cmd = [
        "tmc3",
        "--mode=0",
        f"--uncompressedDataPath={in_file}",
        f"--compressedStreamPath={bin_file}",
        *[f"--{k}={v}" for k, v in kwargs.items()],
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(p.stdout)
    return {}


def tmc3_decompress(bin_file, rec_file, **kwargs):
    cmd = [
        "tmc3",
        "--mode=1",
        f"--compressedStreamPath={bin_file}",
        f"--reconstructedDataPath={rec_file}",
        *[f"--{k}={v}" for k, v in kwargs.items()],
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(p.stdout)
    num_bits = None
    for line in p.stdout.splitlines():
        m = re.match(r"^positions bitstream size (?P<num_bytes>[0-9]+) B$", line)
        if m is not None:
            num_bits = int(m.group("num_bytes")) * 8
            break
    if num_bits is None:
        raise RuntimeError("Could not parse bitstream size")
    return {"num_bits": num_bits}


def write_row_tsv(row, filename, mode):
    row_str = "\t".join(f"{x}" for x in row)
    with open(filename, mode) as f:
        print(row_str, file=f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out_results_tsv", default="input_codec_dataset.tsv")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.out_results_tsv).exists():
        row = ["codec", "in_file", "bin_file", "rec_file", "num_bits"]
        write_row_tsv(row, args.out_results_tsv, "w")

    for in_file in sorted(Path(args.in_dir).rglob("*.ply")):
        relative_path = in_file.relative_to(args.in_dir)
        bin_file = Path(args.out_dir) / relative_path.with_suffix(".bin")
        rec_file = Path(args.out_dir) / relative_path.with_suffix(".rec.ply")
        rec_file.parent.mkdir(parents=True, exist_ok=True)

        out_enc = tmc3_compress(in_file, bin_file, inputScale=str(args.scale))
        out_dec = tmc3_decompress(bin_file, rec_file, outputBinaryPly=0)

        row = [args.codec, in_file, bin_file, rec_file, out_dec["num_bits"]]
        write_row_tsv(row, args.out_results_tsv, "a")

        print(in_file)
        print(bin_file)
        print(rec_file)
        print(f"num_bits: {out_dec['num_bits']}")
        print("-" * 80)


if __name__ == "__main__":
    main()


# NOTE: The by_n_ply directory may be generated generated using ...
