from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


_PLOT_PC_SCATTER_SETTINGS_COMMON = dict(
    x="x",
    y="y",
    z="z",
    color="name",
)

_PLOT_PC_LAYOUT_SETTINGS_COMMON = dict(
    scene=dict(
        xaxis=dict(
            nticks=5,
            range=[-1, 1],
        ),
        yaxis=dict(
            nticks=5,
            range=[-1, 1],
        ),
        zaxis=dict(
            nticks=5,
            range=[-1, 1],
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25),
        ),
    ),
)

PLOT_PC_SETTINGS = {
    "default": {
        "scatter_kwargs": dict(
            **_PLOT_PC_SCATTER_SETTINGS_COMMON,
        ),
        "layout_kwargs": dict(
            **_PLOT_PC_LAYOUT_SETTINGS_COMMON,
        ),
    },
}


def plot_point_cloud(
    df: pd.DataFrame,
    scatter_kwargs: dict[str, Any] = {},
    layout_kwargs: dict[str, Any] = {},
):
    """Plots point cloud."""
    import plotly.express as px

    scatter_kwargs = {**PLOT_PC_SETTINGS["default"]["scatter_kwargs"], **scatter_kwargs}
    layout_kwargs = {**PLOT_PC_SETTINGS["default"]["layout_kwargs"], **layout_kwargs}
    fig = px.scatter_3d(df, **scatter_kwargs)
    fig.update_layout(**layout_kwargs)
    fig.update_traces(marker_size=1)
    return fig
