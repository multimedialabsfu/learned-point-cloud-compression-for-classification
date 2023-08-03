import os
import re

import bjontegaard as bd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src  # noqa: F401
from compressai_trainer.utils.compressai.results import compressai_dataframe

sns.set_theme(
    # context="paper",
    style="whitegrid",
    font="Times New Roman",
    # font_scale=1,
)

EPSILON = 1e-8

DATASET = "ModelNet40"
x = "bit_loss"
y = "acc_top1"

CODEC_TYPES = ["full", "lite", "micro", "input-compression"]

CODECS = [
    "full_points=1024",
    "full_points=512",
    "full_points=256",
    "full_points=128",
    "full_points=64",
    "full_points=32",
    "full_points=16",
    "full_points=8",
    "lite_points=1024",
    "lite_points=512",
    "lite_points=256",
    "lite_points=128",
    "lite_points=64",
    "lite_points=32",
    "lite_points=16",
    "lite_points=8",
    "micro_points=1024",
    "micro_points=512",
    "micro_points=256",
    "micro_points=128",
    "micro_points=64",
    "micro_points=32",
    "micro_points=16",
    "micro_points=8",
    "input-compression-tmc13",
    "input-compression-octattention",
    "input-compression-draco",
]

REF_CODEC_NAME = "Input compression codec [tmc13, P=*]"

COLORS = {
    "baseline_with_transform": "#999999",
    "baseline_no_transform": "#CCCCCC",
}

BD_MAX_BITRATES = {
    "input-compression": {
        "ref": 1e12,
        "curr": 1e12,
    },
    "full": {
        "ref": 2000,
        "curr": 1000,
    },
    "lite": {
        "ref": 800,
        "curr": 200,  # Unnecessary.
    },
    "micro": {
        "ref": 500,
        "curr": 100,  # Unnecessary.
    },
}


def read_dataframe():
    df = pd.concat(
        [
            compressai_dataframe(
                codec_name=name,
                dataset="point-cloud-classification/modelnet40",
                filename_format="{codec_name}",
            )
            for name in CODECS
        ]
    )

    def name_to_codec_type(name):
        name_pattern = r"^Proposed codec \[(?P<codec_type>[\w-]+), P=(?P<points>\d+)\]$"
        m = re.match(name_pattern, name)
        if m:
            return m.group("codec_type")
        return "input-compression"

    df["codec_type"] = df["name"].apply(name_to_codec_type)

    return df


def plot_baseline(ax, dataset):
    if dataset == "ModelNet40":
        _, x_max = ax.get_xlim()
        ax.plot(
            [0, x_max],
            [89.2, 89.2],
            label="PointNet (with transforms) [official] [89.2%]",
            color=COLORS["baseline_with_transform"],
            linestyle="--",
        )
        ax.plot(
            [0, x_max],
            [87.1, 87.1],
            label="PointNet (no transforms) [official] [87.1%]",
            color=COLORS["baseline_no_transform"],
            linestyle="--",
        )


def write_codec_plot(df, codec_type):
    fig, ax = plt.subplots(figsize=(0.9 * 6.4, 1.0 * 4.8))

    if codec_type == "input-compression":
        ax.set(
            xlabel="Rate (bits)",
            ylabel="Top-1 Accuracy (%)",
            xlim=[0, 3000],
            ylim=[0, 100],
        )
        palette = [
            *sns.color_palette("husl", 9)[-1:],
            *sns.color_palette("cubehelix", 3)[:2],
        ]
    else:
        ax.set(
            xlabel="Rate (bits)",
            ylabel="Top-1 Accuracy (%)",
            xlim=[0, 600],
            ylim=[0, 100],
        )
        palette = sns.color_palette("husl", 9)

    plot_baseline(ax, DATASET)

    mask = (df["codec_type"] == codec_type) | (df["name"] == REF_CODEC_NAME)
    # mask |= (df["codec_type"] == "input-compression")
    df_curr = df[mask]

    sns.lineplot(ax=ax, data=df_curr, x=x, y=y, hue="name", palette=palette)
    ax.legend().set_title(None)

    root_dir = "results/plot_rd/pdf/rate_accuracy"
    os.makedirs(root_dir, exist_ok=True)
    fig.savefig(
        f"{root_dir}/{DATASET.lower()}_{codec_type}.pdf",
        bbox_inches="tight",
        # pad_inches=0,
    )
    plt.close(fig)


def preprocess_for_bd(x_ref, y_ref, x_curr, y_curr):
    x_ref = np.array(x_ref)
    y_ref = np.array(y_ref)
    x_curr = np.array(x_curr)
    y_curr = np.array(y_curr)

    # Ensure strictly monotonically increasing.
    for arr in [x_ref, x_curr, y_ref, y_curr]:
        assert (arr[:-1] < arr[1:]).all()

    x_min = min(x_ref[0], x_curr[0])
    x_max = max(x_ref[-1], x_curr[-1])

    # if x_ref[0] > x_min:
    #     x_ref = [x_min, *x_ref]
    #     y_ref = [INTERPOLATE_WITH_RD_ORIGIN, *y_ref]
    #
    # if x_curr[0] > x_min:
    #     x_curr = [x_min, *x_curr]
    #     y_curr = [INTERPOLATE_WITH_RD_ORIGIN, *y_curr]

    if "avoid_x_zero":
        x_ref[x_ref == 0] = x_ref[1] - EPSILON
        x_curr[x_curr == 0] = x_curr[1] - EPSILON

    if x_ref[-1] < x_max:
        x_ref = [*x_ref, x_max]
        y_ref = [*y_ref, y_ref[-1] + EPSILON]

    if x_curr[-1] < x_max:
        x_curr = [*x_curr, x_max]
        y_curr = [*y_curr, y_curr[-1] + EPSILON]

    return x_ref, y_ref, x_curr, y_curr


def compute_stats(name, df_curr, df_ref):
    # codec_type = df_curr["codec_type"].unique()[0]
    # df_curr = df_curr[df_curr["bit_loss"] < BD_MAX_BITRATES[codec_type]["curr"]]
    # df_ref = df_ref[df_ref["bit_loss"] < BD_MAX_BITRATES[codec_type]["ref"]]

    x_ref, y_ref, x_curr, y_curr = preprocess_for_bd(
        df_ref[x], df_ref[y], df_curr[x], df_curr[y]
    )

    bd_rate = bd.bd_rate(
        rate_anchor=x_ref,
        dist_anchor=y_ref,
        rate_test=x_curr,
        dist_test=y_curr,
        method="akima",
        require_matching_points=False,
    )

    bd_dist = bd.bd_psnr(
        rate_anchor=x_ref,
        dist_anchor=y_ref,
        rate_test=x_curr,
        dist_test=y_curr,
        method="akima",
        require_matching_points=False,
    )

    max_y = df_curr[y].max()

    print(f"{name:<40} & {max_y:6.1f} & {bd_rate:6.1f} & {bd_dist:6.1f} \\\\")


def main():
    df = read_dataframe()
    print(df.to_string())

    df_ref = df[df["name"] == REF_CODEC_NAME]

    for name, df_curr in df.groupby("name", sort=False):
        compute_stats(name, df_curr, df_ref)

    for codec_type in CODEC_TYPES:
        write_codec_plot(df, codec_type)


if __name__ == "__main__":
    main()
