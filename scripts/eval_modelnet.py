from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from catalyst import metrics
from compressai.datasets.modelnet import PointCloudPreprocessor
from compressai.registry import register_dataset
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from compressai_trainer.run.eval_model import setup
from compressai_trainer.utils.metrics import compute_metrics
from compressai_trainer.utils.point_cloud import pc_read

FILESIZES_TSV = "results/tmc13_file_sizes.tsv"
AGGREGATE_TSV = "results/tmc13_pointnet_001_aggregate.tsv"
AGGREGATE_NEW_TSV = "results/tmc13_pointnet_001_aggregate_new.tsv"
SAMPLES_TSV = "results/tmc13_pointnet_001_samples.tsv"
SAMPLES_NEW_TSV = "results/tmc13_pointnet_001_samples_new.tsv"


# Various scale parameters (similar to "QP") that dataset was encoded using.
SCALES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# PointNet varying N models.
RUN_HASHES = [
    "f84aa21a1ce3491d800733a6",  # 8
    "552499a134ac41cebc50e967",  # 16
    "d19f6ea297bb42b1902c4678",  # 32
    "010feb1990e44251ad080a35",  # 64
    "bad325156b3e4137bc7752d9",  # 128
    "e2f872d550c548f2b687c88a",  # 256
    "5d0ef9a2a12b4752b11cd61b",  # 512
    "b8d279f32b3f43d89d386b11",  # 1024
    "e3eeddad2cc64348b4f03c89",  # 2048
]

MIN_SCALES = [
    1,  # 8
    1,  # 16
    1,  # 32
    1,  # 64
    1,  # 128
    1,  # 256
    1,  # 512
    1,  # 1024
    1,  # 2048
]


@register_dataset("PlyFolderDataset")
class PlyFolderDataset(Dataset):
    """Dataset for point-clouds."""

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.paths = sorted(Path(root).glob("**/*.ply"))
        self.transform = transform
        self.preprocessor = PointCloudPreprocessor()

    def __getitem__(self, index):
        path = self.paths[index]
        pattern = r"^(?P<loader>.*)_(?P<label_idx>\d+)_(?P<index>\d+)(\.bin)?\.ply$"
        match = re.match(pattern, path.name)
        if match is None:
            raise ValueError(f"Could not parse path: {path}")
        label_idx = int(match.group("label_idx"))
        points = pc_read(path)
        # assert points.shape[0] == self.num_points
        assert points.shape[1] == 3
        points = self.preprocessor.preprocess(points)
        points = self.transform(points)
        return {
            "index": index,
            "points": points,
            "labels": label_idx,
            # "path": str(path),
        }

    def __len__(self):
        return len(self.paths)


# TODO Document how to load a model/runner from a run_hash/config, etc... always confuses me.
def load_by_run_hash(run_hash, scale):
    path = f"/home/mulhaq/data/runs/pc-mordor/{run_hash}/configs/config.yaml"
    conf = OmegaConf.load(path)
    del conf.dataset["train"]
    del conf.dataset["valid"]
    conf.dataset["infer"] = {
        "type": "PlyFolderDataset",
        "config": {
            "root": f"by_n_scale_ply/{conf.hp.num_points:04}/{scale:04}/modelnet40",
            # "split": "infer",
            # "num_points": conf.hp.num_points,
        },
        "loader": {
            "shuffle": False,
            # "batch_size": 32,
            "batch_size": 1,
            "num_workers": 4,
        },
        "settings": {},
        "transforms": [],
        "meta": {
            "name": "ModelNet40",
            "identifier": None,
            "num_samples": 2468,
            "steps_per_epoch": 0,
        },
    }
    conf.model.source = "config"
    conf.paths.model_checkpoint = "${paths.checkpoints}/runner.last.pth"
    runner = setup(conf)
    return runner


def read_filesizes(path):
    df = pd.read_csv(path, sep="\t")
    df_split = df["binfile"].str.extract(
        r"^by_n_scale_ply/"
        r"(?P<num_points_str>\d+)/"
        r"(?P<scale_str>\d+)/"
        r"(?P<dataset>[^/]+)/"
        r"infer_(?P<label_str>\d+)_(?P<sample_index_str>\d+)\.bin$"
    )
    df_split["num_points"] = df_split["num_points_str"].astype(int)
    df_split["scale"] = df_split["scale_str"].astype(int)
    df_split["label"] = df_split["label_str"].astype(int)
    df_split["sample_index"] = df_split["sample_index_str"].astype(int)
    df_split["filesize_bits"] = df["size"] * 8
    df_split.drop(
        ["num_points_str", "scale_str", "label_str", "sample_index_str"],
        axis=1,
        inplace=True,
    )
    df_split.sort_values(["num_points", "scale", "label", "sample_index"], inplace=True)
    df_split.reset_index(drop=True, inplace=True)
    df = df_split
    return df


def read_samples(path):
    df = pd.read_csv(path, sep="\t")
    df.sort_values(["num_points", "scale", "label", "dataset_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    x = df["label"]
    b = np.diff(x, prepend=[x[0] - 1]) != 0
    df["sample_index"] = arange_at_edges(b)
    df.sort_values(["num_points", "scale", "label", "sample_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def join_tsvs():
    df1 = read_samples(SAMPLES_TSV)
    df2 = read_filesizes(FILESIZES_TSV)

    keys = ["num_points", "scale", "label", "sample_index"]
    df = pd.merge(df1, df2, how="left", on=keys)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(SAMPLES_NEW_TSV, sep="\t", index=False)
    return df


def arange_at_edges(b, dtype=int):
    [nz] = b.nonzero()
    b_ = np.ones(b.shape, dtype=dtype)
    b_[0] = 0
    b_[nz[1:]] += nz[:-1] - nz[1:]
    return b_.cumsum()


def run(run_hash, scale):
    runner = load_by_run_hash(run_hash, scale)
    loader = runner.loaders["infer"]

    num_points = runner.hparams["hp"]["num_points"]
    metrics_ = runner.hparams["runner"]["metrics"]
    meters_ = runner.hparams["runner"]["meters"]["infer"]
    meters = {key: metrics.AdditiveMetric(compute_on_call=False) for key in meters_}

    for batch in loader:
        batch_size = len(batch["index"])
        batch = {k: v.to(runner.engine.device) for k, v in batch.items()}

        out_infer = runner.predict_batch(batch)
        out_net = out_infer["out_net"]
        out_metrics = compute_metrics(out_net, batch, metrics_)
        print(out_net["t_hat"].argmax().item(), batch["labels"].item())

        for key in out_metrics.keys():
            if key not in meters:
                continue
            meters[key].update(out_metrics[key], batch_size)

        metrics_str = "\t".join(
            f"{v:.6f}" for k, v in out_metrics.items() if k in meters
        )
        with open(SAMPLES_TSV, "a") as f:
            print(
                f"{run_hash}\t{num_points}\t{scale}\t"
                f"{batch['labels'].item()}\t"
                f"{batch['index'].item()}\t"
                f"{batch['points'].shape[1]}\t"
                f"{out_net['t_hat'].argmax().item()}\t"
                f"{metrics_str}",
                file=f,
            )

    print({k: m.mean for k, m in meters.items()})

    metrics_str = "\t".join(f"{v.mean:.6f}" for _, v in meters.items())

    with open(AGGREGATE_TSV, "a") as f:
        print(f"{run_hash}\t{num_points}\t{scale}\t{metrics_str}", file=f)


def run_evals():
    meters_str = "\t".join(["acc_top1", "acc_top3"])
    with open(SAMPLES_TSV, "a") as f:
        print(
            f"run_hash\tnum_points\tscale\tlabel\tdataset_index\t"
            f"num_points_sample\tpred_label\t{meters_str}",
            file=f,
        )
    with open(AGGREGATE_TSV, "a") as f:
        print(f"run_hash\tnum_points\tscale\t{meters_str}", file=f)

    for run_hash, min_scale in zip(RUN_HASHES, MIN_SCALES):
        for scale in SCALES:
            if scale < min_scale:
                continue
            run(run_hash, scale)


def run_analysis():
    df = join_tsvs()

    df_agg = (
        df.groupby(["num_points", "scale"])[["filesize_bits", "acc_top1", "acc_top3"]]
        .mean()
        .reset_index()
    )
    df_agg.to_csv(AGGREGATE_NEW_TSV, sep="\t", index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    sns.scatterplot(df_agg, x="filesize_bits", y="acc_top1", hue="num_points", ax=ax)

    plt.show()
    fig.savefig(
        "results/plot_rd/rate_accuracy_tmc13_pointnet_001_acc_vs_size_all_samples.png",
        dpi=300,
    )


def main():
    # run_evals()
    run_analysis()


if __name__ == "__main__":
    main()
