from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import torch

from src.utils.metrics import compute_metrics
from src.utils.point_cloud import pc_read

OCT_ATTENTION_VIRTUALENV_DIR = (
    "/home/mulhaq/.cache/pypoetry/virtualenvs/octattention-4URgkG2e-py3.11"
)
OCT_ATTENTION_DIR = (
    "/mnt/data/code/research/downloaded/point-cloud-compression/OctAttention"
)


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


def draco_compress(in_file, bin_file, **kwargs):
    cmd = [
        "draco_encoder",
        "-point_cloud",
        "-i",
        f"{in_file}",
        "-o",
        f"{bin_file}",
        *[x for k, v in kwargs.items() for x in [f"-{k}", f"{v}"]],
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(p.stdout)
    num_bits = None
    for line in p.stdout.splitlines():
        m = re.match(r"^Encoded size = (?P<num_bytes>[0-9]+) bytes$", line)
        if m is not None:
            num_bits = int(m.group("num_bytes")) * 8
            break
    if num_bits is None:
        raise RuntimeError("Could not parse bitstream size")
    return {"num_bits": num_bits}


def draco_decompress(bin_file, rec_file, **kwargs):
    cmd = [
        "draco_decoder",
        "-point_cloud",
        "-i",
        f"{bin_file}",
        "-o",
        f"{rec_file}",
        *[x for k, v in kwargs.items() for x in [f"-{k}", f"{v}"]],
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(p.stdout)
    return {}


def octattention_compress_decompress(in_file, bin_file, rec_file, **kwargs):
    cmd = [
        f"{OCT_ATTENTION_VIRTUALENV_DIR}/bin/python",
        f"{OCT_ATTENTION_DIR}/run_codec.py",
        f"--ckpt_path={OCT_ATTENTION_DIR}/modelsave/obj/encoder_epoch_00800093.pth",
        f"--in_file={in_file}",
        f"--bin_file={bin_file}",
        f"--rec_file={rec_file}",
        *[f"--{k}={v}" for k, v in kwargs.items()],
    ]
    try:
        p = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=OCT_ATTENTION_DIR
        )
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e
    num_bits = None
    for line in p.stdout.splitlines():
        m = re.match(r"^binsize\(b\): (?P<num_bits>[0-9]+)$", line)
        if m is not None:
            num_bits = int(m.group("num_bits"))
            break
    if num_bits is None:
        raise RuntimeError("Could not parse bitstream size")
    out = {"num_bits": num_bits}
    return {"out_enc": out, "out_dec": out}


# NOTE: This codec is run externally beforehand.
def external_compress_decompress(in_file, bin_file, rec_file, **kwargs):
    common_file = Path(bin_file).with_suffix("")
    bin_files = sorted(common_file.parent.glob(f"{common_file.name}.*bin"))
    assert len(bin_files) > 0
    num_bits = sum(sub_bin_file.stat().st_size * 8 for sub_bin_file in bin_files)
    out = {"num_bits": num_bits}
    return {"out_enc": out, "out_dec": out}


def run_codec(codec, in_file, bin_file, rec_file, scale, **kwargs):
    if codec == "tmc3":
        out_enc = tmc3_compress(in_file, bin_file, inputScale=scale, **kwargs)
        out_dec = tmc3_decompress(bin_file, rec_file, outputBinaryPly=0)
        return {"out_enc": out_enc, "out_dec": out_dec}
    if codec == "draco":
        qp = math.log2(scale)
        out_enc = draco_compress(in_file, bin_file, qp=qp, cl=10, **kwargs)
        out_dec = draco_decompress(bin_file, rec_file)
        return {"out_enc": out_enc, "out_dec": out_dec}
    if codec == "octattention":
        for bptt in [64, 32, 16]:
            try:
                return octattention_compress_decompress(
                    in_file, bin_file, rec_file, scale=scale, bptt=bptt, **kwargs
                )
            except subprocess.CalledProcessError:
                pass
        raise RuntimeError("Could not compress with OctAttention")
    if codec == "external":
        return external_compress_decompress(in_file, bin_file, rec_file, **kwargs)
    raise ValueError(f"Unknown codec: {codec}")


def write_row_tsv(row, filename, mode):
    row_str = "\t".join(f"{x}" for x in row)
    with open(filename, mode) as f:
        print(row_str, file=f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out_results_tsv", default="input_codec_dataset.tsv")
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.out_results_tsv).exists():
        row = [
            "codec",
            "in_file",
            "ref_file",
            "bin_file",
            "rec_file",
            "num_bits",
            "rec_loss",
            "d1-psnr",
        ]
        write_row_tsv(row, args.out_results_tsv, "w")

    for in_file in sorted(Path(args.in_dir).rglob("*.ply")):
        relative_path = in_file.relative_to(args.in_dir)
        ref_file = Path(args.ref_dir) / relative_path.with_suffix(".ply")
        bin_file = Path(args.out_dir) / relative_path.with_suffix(".bin")
        rec_file = Path(args.out_dir) / relative_path.with_suffix(".rec.ply")
        rec_file.parent.mkdir(parents=True, exist_ok=True)

        out = run_codec(args.codec, in_file, bin_file, rec_file, args.scale)
        num_bits = (
            out["out_enc"]["num_bits"]
            if "num_bits" in out["out_enc"]
            else out["out_dec"]["num_bits"]
        )

        scale_factor = args.scale if args.codec == "tmc3" else 1.0
        x = {"points": torch.from_numpy(pc_read(ref_file))[None, ...]}
        x_hat = {"x_hat": torch.from_numpy(pc_read(rec_file))[None, ...] / scale_factor}
        assert x_hat["x_hat"].abs().max() <= 2.0
        metrics = compute_metrics(x, x_hat, ["chamfer", "d1-psnr"])

        row = [
            args.codec,
            in_file,
            ref_file,
            bin_file,
            rec_file,
            num_bits,
            metrics["chamfer"],
            metrics["d1-psnr"],
        ]
        write_row_tsv(row, args.out_results_tsv, "a")

        print(in_file)
        print(ref_file)
        print(bin_file)
        print(rec_file)
        print(f"num_bits: {num_bits}")
        print(f"rec_loss: {metrics['chamfer']}")
        print(f"d1-psnr: {metrics['d1-psnr']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
