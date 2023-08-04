import json

import aim

from compressai_trainer.run.plot_rd import HOVER_METRICS
from compressai_trainer.utils.aim.query import get_runs_dataframe, run_hashes_by_query
from compressai_trainer.utils.optimal import optimal_dataframe

AIM_REPO_PATH = "/home/mulhaq/data/aim/pc-mordor/pcc"

OPTIMAL_METHOD = "convex"

X = "bit_loss"
Y = "acc_top1"

HOVER_HPARAMS_COMMON = [
    "model.name",
    "criterion.lmbda.cls",
    "dataset.train.meta.name",
    "hp.num_classes",
    "hp.num_points",
]

HOVER_HPARAMS_FULL = [
    *HOVER_HPARAMS_COMMON,
    "hp.num_channels.g_a.pointwise",
    "hp.num_channels.g_a.mixer",
    "hp.num_channels.task_backend",
    "hp.groups.g_a.pointwise",
]

HOVER_HPARAMS_LITE = [
    *HOVER_HPARAMS_FULL,
]

HOVER_HPARAMS_MICRO = [
    *HOVER_HPARAMS_COMMON,
    "hp.num_channels.g_a.pointwise",
    "hp.num_channels.g_a.mixer",
    "hp.num_channels.task_backend.transform.pointwise",
    "hp.num_channels.task_backend.transform.mixer",
    "hp.num_channels.task_backend.mlp",
    "hp.groups.g_a.pointwise",
    "hp.groups.task_backend.transform.pointwise",
]

VARYING_HPARAMS = [
    "criterion.lmbda.cls",
]


def validate_dataframe(df, hover_hparams):
    assert df["name"].nunique() == 1
    assert df["model.name"].nunique() == 1
    assert df["experiment"].nunique() == 1

    for column in hover_hparams:
        if column in VARYING_HPARAMS:
            continue
        if df[column].dtype == "object":
            if isinstance(df[column].iloc[0], list):
                assert df[column].map(tuple).nunique() == 1
            else:
                assert df[column].nunique() == 1
        else:
            assert df[column].nunique() == 1


def write_json(df, path, hover_hparams):
    exclude_columns = ["name", "experiment", "model.name", *hover_hparams]
    results_columns = [X, Y, "run_hash", *VARYING_HPARAMS, "epoch", *HOVER_METRICS]
    results_columns += [
        c for c in df.columns if c not in results_columns and c not in exclude_columns
    ]

    exclude_columns = [*VARYING_HPARAMS]
    meta_columns = [c for c in hover_hparams if c not in exclude_columns]

    out_dict = {
        "name": df["name"].unique()[0],
        "description": "",
        "meta": df[meta_columns].iloc[0].to_dict(),
        "results": {
            k: list(v.values()) for k, v in df[results_columns].to_dict().items()
        },
    }

    with open(path, "w") as f:
        json.dump(out_dict, f, indent=2)


def run_writer(repo, meta, name, hover_hparams):
    query = " and ".join(f"run.{k} == {repr(v)}" for k, v in meta.items())

    stem_keys = ["model.name", "hp.num_channels.g_a.pointwise", "hp.num_points"]
    stem = ";".join(f"{k}={meta[k]}" for k in stem_keys)
    out_path = f"results/plot_rd/json/{stem}.json"

    print(f"Query: {query}")
    print(f"Output path: {out_path}")

    run_hashes = run_hashes_by_query(repo, query)
    assert len(run_hashes) > 0

    df = get_runs_dataframe(
        run_hashes=run_hashes,
        repo=repo,
        metrics=[X, Y, *HOVER_METRICS, "bpp_loss"],
        hparams=hover_hparams,
        epoch="best",
    )

    validate_dataframe(df, hover_hparams)

    assert X == "bit_loss" and Y == "acc_top1"
    df["acc_top1"] = df["acc_top1"] * 100
    df["bit_loss"] = df["bpp_loss"]
    df["name"] = name
    df.drop(columns=["bpp_loss"], inplace=True)

    df = optimal_dataframe(df, x=X, y=Y, method=OPTIMAL_METHOD)

    write_json(df, out_path, hover_hparams)


def main():
    repo = aim.Repo(AIM_REPO_PATH)

    for num_points in [1024, 512, 256, 128, 64, 32, 16, 8]:
        meta = {
            "model.name": "um-pcc-cls-only-pointnet",
            "dataset.train.meta.name": "ModelNet40",
            "hp.num_classes": 40,
            "hp.num_channels.g_a.pointwise": [3, 64, 64, 64, 128, 1024],
            "hp.num_channels.g_a.mixer": [],
            "hp.num_channels.task_backend": [1024, 512, 256, 40],
            "hp.groups.g_a.pointwise": [1, 1, 1, 1, 1],
            "hp.num_points": num_points,
        }

        run_writer(
            repo,
            meta,
            name=f"Proposed codec [full, P={num_points}]",
            hover_hparams=HOVER_HPARAMS_FULL,
        )

    for num_points in [1024, 512, 256, 128, 64, 32, 16, 8]:
        meta = {
            "model.name": "um-pcc-cls-only-pointnet",
            "dataset.train.meta.name": "ModelNet40",
            "hp.num_classes": 40,
            "hp.num_channels.g_a.pointwise": [3, 8, 8, 16, 16, 32],
            "hp.num_channels.g_a.mixer": [],
            "hp.num_channels.task_backend": [32, 512, 256, 40],
            "hp.groups.g_a.pointwise": [1, 1, 1, 2, 4],
            "hp.num_points": num_points,
        }

        run_writer(
            repo,
            meta,
            name=f"Proposed codec [lite, P={num_points}]",
            hover_hparams=HOVER_HPARAMS_LITE,
        )

    for num_points in [1024, 512, 256, 128, 64, 32, 16, 8]:
        meta = {
            "model.name": "um-pcc-cls-only-pointnet-mmsp2023",
            "dataset.train.meta.name": "ModelNet40",
            "hp.num_classes": 40,
            "hp.num_channels.g_a.pointwise": [3, 16],
            "hp.num_channels.g_a.mixer": [],
            "hp.num_channels.task_backend.transform.pointwise": [16, 64, 64, 64],
            "hp.num_channels.task_backend.transform.mixer": [],
            "hp.num_channels.task_backend.mlp": [64, 512, 256, 40],
            "hp.groups.g_a.pointwise": [1],
            "hp.groups.task_backend.transform.pointwise": [1, 1, 1],
            "hp.num_points": num_points,
        }

        run_writer(
            repo,
            meta,
            name=f"Proposed codec [micro, P={num_points}]",
            hover_hparams=HOVER_HPARAMS_MICRO,
        )


if __name__ == "__main__":
    main()
