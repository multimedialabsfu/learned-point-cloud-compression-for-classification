import sys

import aim
import matplotlib.pyplot as plt
import pandas as pd
from aim.storage.context import Context

from compressai_trainer.utils.aim.query import _close_run, run_hashes_by_query

AIM_REPO_PATH = "/home/mulhaq/data/aim/pc-mordor/pcc"
OUT_DIR = "results/plot_pointcloud/pdf"


def dataframe_from_plotly_figure(fig):
    records = [
        {"x": trace.x[i], "y": trace.y[i], "z": trace.z[i], "name": trace.name}
        for trace in fig.data
        for i in range(len(trace.x))
    ]
    return pd.DataFrame.from_records(records)


def update_plotly_figure(fig, scale):
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-scale, y=-scale, z=scale),
            ),
        ),
    )


def write_updated_plotly_figure(path, fig):
    update_plotly_figure(fig, scale=0.6)
    fig.write_image(path, scale=2)


def write_mpl_figure(path, df, key):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    style = dict(marker=".", s=32.0)

    for name, group in df.groupby("name"):
        if key not in name:
            continue
        ax.scatter(group["x"], group["y"], group["z"], label=name, **style)

    # Draw outer box points for reference.
    # ax.scatter(
    #     [-1, -1, -1, -1, 1, 1, 1, 1],
    #     [-1, -1, 1, 1, -1, -1, 1, 1],
    #     [-1, 1, -1, 1, -1, 1, -1, 1],
    #     label="endpoints",
    # )

    set_mpl_camera_scale(ax, scale=1.05)
    ax.view_init(elev=45, azim=-145, roll=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def set_mpl_camera_scale(ax, scale=1):
    ax.set_xlim3d(-scale / 2, scale / 2)
    ax.set_zlim3d(-scale / 2, scale / 2)
    ax.set_ylim3d(-scale / 2, scale / 2)


def get_run_figures(run_hash, repo):
    run = aim.Run(run_hash, repo=repo, read_only=True)
    seq = run.get_figure_sequence("point-cloud", Context({"name": "x_hat"}))
    idxs, (figures, _, _) = seq.data.items_list()
    _close_run(run)

    figures = [x.to_plotly_figure() for x in figures]
    figures_dict = dict(zip(idxs, figures))
    return figures_dict


def main():
    repo = aim.Repo(AIM_REPO_PATH)
    # run_hashes = run_hashes_by_query(repo, query)
    run_hashes = sys.argv[1:]

    for i, run_hash in enumerate(run_hashes):
        figures_dict = get_run_figures(run_hash, repo)

        for fig_idx in [1, 4]:
            df = dataframe_from_plotly_figure(figures_dict[fig_idx])

            # Write reference figures, too.
            if i == 0:
                write_mpl_figure(f"{OUT_DIR}/{fig_idx}_ref.pdf", df, key="(x)")

            write_mpl_figure(f"{OUT_DIR}/{fig_idx}_{i:02}.pdf", df, key="(x_hat)")


if __name__ == "__main__":
    main()
