from __future__ import annotations

import pandas as pd

# isort: off

# Import this first, since monkey-patching is incompatible...
import compressai_trainer.run.plot_rd as _M

# NOTE: This will be monkey-patched *later* in the imports, but we need to
# run some init code using the patched version, so we import it instead of
# from compressai_trainer.utils.compressai.results import compressai_dataframe
from src.utils.compressai.results import compressai_dataframe

# isort: on

from compressai_trainer.plot import plot_rd

_M.DATASET = "point-cloud/classification/modelnet40"
_M.TITLE = "Performance evaluation on ModelNet40 - Top-1 Accuracy"

# _M.METRIC = "acc_top1"
_M.METRIC = "none"
_M.PLOT_RD_METRIC = "acc_top1"

_M.PLOT_RD_SETTINGS = {
    "acc_top1": {
        "scatter_kwargs": dict(
            # x="bit_loss",
            # y="acc_top1",
            color="name",
        ),
        "layout_kwargs": dict(
            xaxis_title="Rate (bits)",
            xaxis=dict(range=[0.0, 250.0], tick0=0.0, dtick=25.0),
            yaxis_title="Top-1 Accuracy",
            yaxis=dict(range=[0.0, 1.0], tick0=0.0, dtick=5),
        ),
    },
}

_M.COMPRESSAI_CODECS = [
    # Point-cloud codecs:
    "theoretical-optimum",
    "sfu-pcc-cls-only-pointnet_size=full_points=1024",
    "sfu-pcc-cls-only-pointnet_size=lite_points=1024",
    "sfu-pcc-cls-only-pointnet_size=lite_points=512",
    "sfu-pcc-cls-only-pointnet_size=lite_points=256",
    "sfu-pcc-cls-only-pointnet_size=lite_points=128",
    "sfu-pcc-cls-only-pointnet_size=lite_points=64",
    "sfu-pcc-cls-only-pointnet_size=lite_points=32",
    "sfu-pcc-cls-only-pointnet_size=lite_points=16",
    "sfu-pcc-cls-only-pointnet_size=lite_points=8",
    "input-compression-pointnet-tmc13",
]


_M.HOVER_HPARAMS = [
    "model.name",
    # "criterion.lmbda",
    "criterion.lmbda.cls",
    # "criterion.lmbda.rec",
    "dataset.train.meta.name",
    "hp.num_classes",
    "hp.num_points",
    #
    # Full:
    # "hp.num_channels.g_a",
    # "hp.num_channels.task_backend",
    # "hp.groups.g_a",
    #
    # Mini:
    "hp.num_channels.g_a",
    "hp.num_channels.task_backend",
    "hp.groups.g_a",
    #
]

_M.HOVER_METRICS = [
    "loss",
]

_M.HOVER_DATA = [
    "run_hash",
    "name",
    "model.name",
    "experiment",
    "epoch",
]

_M.HOVER_DATA += _M.HOVER_HPARAMS + _M.HOVER_METRICS


def _reference_dataframes():
    dfs = [
        compressai_dataframe(
            codec_name=name,
            dataset=_M.DATASET,
            filename_format="{codec_name}",
        )
        for name in _M.COMPRESSAI_CODECS
    ]

    for df in dfs:
        df["acc_top1"] = df["acc_top1"] / 100.0
        df["bpp_loss"] = df["bit_loss"]

    return dfs


# For compressai_trainer <= 0.3.9:
__ref_dfs = _reference_dataframes()
_M.REFERENCE_DF = pd.concat(__ref_dfs) if __ref_dfs else None


def plot_dataframe(df: pd.DataFrame, args):
    scatter_kwargs = dict(
        **_M.PLOT_RD_SETTINGS[_M.PLOT_RD_METRIC]["scatter_kwargs"],
        x=args.x,
        y=args.y,
        hover_data=_M.HOVER_DATA,
    )

    layout_kwargs = {
        **_M.PLOT_RD_SETTINGS[_M.PLOT_RD_METRIC]["layout_kwargs"],
        "title": _M.TITLE,
    }

    print(df)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    fig = plot_rd(
        df,
        scatter_kwargs=scatter_kwargs,
        layout_kwargs=layout_kwargs,
        metric=_M.METRIC,
    )

    if args.out_html:
        from plotly.offline import plot

        plot(fig, auto_open=False, filename=args.out_html)

    if args.show:
        fig.show()


# Monkey patch:
_M.plot_dataframe = plot_dataframe
