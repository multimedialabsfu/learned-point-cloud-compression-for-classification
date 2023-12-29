from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catalyst.metrics import AdditiveMetric
from omegaconf import OmegaConf

import src.models  # noqa: F401
from compressai_trainer.run.eval_model import setup
from compressai_trainer.utils.metrics import compute_metrics

RESULTS_HEADERS = ["in_file", "bin_file", "rec_file", "num_bits"]


def setup_config(conf, args):
    del conf.dataset["train"]
    del conf.dataset["valid"]
    # TODO(refactor): Extract this to YAML config.
    conf.dataset["infer"] = {
        "type": "PlyFolderDataset",
        "config": {
            "root": args.in_dir,
            # "split": "infer",
            # "num_points": conf.hp.num_points,
        },
        "loader": {
            "shuffle": False,
            "batch_size": 1,
            "num_workers": 4,
        },
        "settings": {},
        "transforms": [],
        "meta": {
            "name": "ModelNet40",
            "identifier": None,
            "num_samples": 2468,  # NOTE: Not necessarily accurate...
            "steps_per_epoch": 0,
        },
    }
    conf.model.source = "config"
    conf.paths.model_checkpoint = "${paths.checkpoints}/runner.last.pth"
    return conf


def write_row_tsv(row, filename, mode):
    row_str = "\t".join(f"{x}" for x in row)
    with open(filename, mode) as f:
        print(row_str, file=f)


def write_tsv_headers(runner, args):
    meters_keys = runner.hparams["runner"]["meters"]["infer"]

    if not Path(args.out_samples_tsv).exists():
        row = [
            "codec",
            "run_hash",
            "num_points",
            "num_points_sample",
            "scale",
            "label",
            "dataset_index",
            "pred_label",
            *RESULTS_HEADERS,
            *meters_keys,
        ]
        write_row_tsv(row, args.out_samples_tsv, "w")

    if not Path(args.out_aggregate_tsv).exists():
        row = [
            "codec",
            "run_hash",
            "num_points",
            "scale",
            "num_bits",
            *meters_keys,
        ]
        write_row_tsv(row, args.out_aggregate_tsv, "w")


def run(runner, args, df_results):
    df_idx = {v: i for i, v in enumerate(df_results["rec_file"])}

    run_hash = runner.hparams["env"]["aim"]["run_hash"]
    num_points = runner.hparams["hp"]["num_points"]
    metrics_keys = runner.hparams["runner"]["metrics"]
    meters_keys = runner.hparams["runner"]["meters"]["infer"]
    meters = {
        key: AdditiveMetric(compute_on_call=False) for key in ["num_bits", *meters_keys]
    }
    loader = runner.loaders["infer"]

    for batch in loader:
        batch_size = len(batch["index"])
        assert batch_size == 1
        batch = {k: v.to(runner.engine.device) for k, v in batch.items()}
        num_points_sample = batch["pos"].shape[1]
        rec_file = loader.dataset.paths[batch["index"].item()]
        print(rec_file)

        df_results_row_idx = df_idx[str(rec_file)]
        df_results_row = df_results.iloc[df_results_row_idx].to_dict()

        out_infer = runner.predict_batch(batch)
        out_metrics = compute_metrics(batch, out_infer["out_net"], metrics_keys)
        out_metrics["num_bits"] = df_results_row["num_bits"]
        label_hat_idx = out_infer["out_net"]["t_hat"].argmax(-1).item()

        for key in out_metrics.keys():
            if key not in meters:
                continue
            meters[key].update(out_metrics[key], batch_size)

        row = [
            args.codec,
            run_hash,
            num_points,
            num_points_sample,
            args.scale,
            batch["label"].item(),
            batch["index"].item(),
            label_hat_idx,
            *[df_results_row[k] for k in RESULTS_HEADERS],
            *[
                f"{out_metrics[k]:.6f}" if k in out_metrics else "nan"
                for k in meters_keys
            ],
        ]
        write_row_tsv(row, args.out_samples_tsv, "a")

    row = [
        args.codec,
        run_hash,
        num_points,
        args.scale,
        f"{meters['num_bits'].mean:.6f}",
        *[f"{meters[k].mean:.6f}" for k in meters_keys],
    ]
    write_row_tsv(row, args.out_aggregate_tsv, "a")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--in_results_tsv", default="input_codec_dataset.tsv")
    parser.add_argument("--out_samples_tsv", default="input_codec_samples.tsv")
    parser.add_argument("--out_aggregate_tsv", default="input_codec_aggregate.tsv")
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    conf = OmegaConf.load(f"{args.config_path}/{args.config_name}.yaml")
    conf = setup_config(conf, args)
    runner = setup(conf)

    df_results = pd.read_csv(args.in_results_tsv, sep="\t")
    write_tsv_headers(runner, args)
    run(runner, args, df_results)


if __name__ == "__main__":
    main()
