import re

import bjontegaard as bd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compressai_trainer.utils.compressai.results import compressai_dataframe

sns.set_theme(
    # context="paper",
    style="whitegrid",
    font="Times New Roman",
    # font_scale=1,
)

DATASET = "ModelNet40"
x = "bit_loss"
y = "acc_top1"

CODEC_TYPES = ["full", "lite", "micro"]

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
]

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

    ax.set(
        xlabel="Rate (bits)",
        ylabel="Top-1 Accuracy (%)",
        xlim=[0, 600],
        ylim=[0, 100],
    )

    plot_baseline(ax, DATASET)

    mask = (df["codec_type"] == codec_type) | (df["codec_type"] == "input-compression")
    df_curr = df[mask]

    sns.lineplot(
        ax=ax,
        data=df_curr,
        x=x,
        y=y,
        hue="name",
        palette=sns.color_palette("husl", 9),
    )
    ax.legend().set_title(None)

    fig.savefig(
        f"results/plot_rd/pdf/rate_accuracy/{DATASET.lower()}_{codec_type}.pdf",
        bbox_inches="tight",
        # pad_inches=0,
    )
    plt.close(fig)


def compute_stats(name, df_curr, df_ref):
    # codec_type = df_curr["codec_type"].unique()[0]
    # df_curr = df_curr[df_curr["bit_loss"] < BD_MAX_BITRATES[codec_type]["curr"]]
    # df_ref = df_ref[df_ref["bit_loss"] < BD_MAX_BITRATES[codec_type]["ref"]]

    bd_rate = bd.bd_rate(
        rate_anchor=df_ref[x],
        dist_anchor=df_ref[y],
        rate_test=df_curr[x],
        dist_test=df_curr[y],
        method="akima",
        require_matching_points=False,
    )

    bd_dist = bd.bd_psnr(
        rate_anchor=df_ref[x],
        dist_anchor=df_ref[y],
        rate_test=df_curr[x],
        dist_test=df_curr[y],
        method="akima",
        require_matching_points=False,
    )

    max_y = df_curr[y].max()

    print(f"{name:<40} & {max_y:6.1f} & {bd_rate:6.1f} & {bd_dist:6.1f}")


def main():
    df = read_dataframe()
    print(df.to_string())

    df_ref = df[df["codec_type"] == "input-compression"]

    for name, df_curr in df.groupby("name", sort=False):
        compute_stats(name, df_curr, df_ref)

    for codec_type in CODEC_TYPES:
        write_codec_plot(df, codec_type)


if __name__ == "__main__":
    main()
