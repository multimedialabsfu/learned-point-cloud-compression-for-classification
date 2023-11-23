from __future__ import annotations

import pandas as pd
from PIL import Image

from compressai_trainer.plot.featuremap import DEFAULT_COLORMAP, featuremap_image
from compressai_trainer.runners.utils import (
    ChannelwiseBppMeter,
    DebugOutputsLogger,
    EbDistributionsFigureLogger,
    GradientClipper,
    RdFigureLogger,
)
from src.plot.point_cloud import plot_point_cloud

__all__ = [
    "ChannelwiseBppMeter",
    "DebugOutputsLogger",
    "EbDistributionsFigureLogger",
    "GradientClipper",
    "RdFigureLogger",
    "PcChannelwiseBppMeter",
    "PcDebugOutputsLogger",
    "PcFigureLogger",
    "point_cloud_dataframe",
]


class PcChannelwiseBppMeter(ChannelwiseBppMeter):
    """Log channel-wise rates (bpp)."""

    def update(self, out_net, input):
        _, P, _ = input["points"].shape
        chan_rate = {
            k: lh.detach().log2().sum(axis=-1) / -P
            for k, lh in out_net["likelihoods"].items()
        }
        try:
            _chan_rate = self._chan_rate  # compressai_trainer>=0.3.11
        except AttributeError:
            _chan_rate = self._chan_bpp  # compressai_trainer<=0.3.10
        for name, ch_rate in chan_rate.items():
            _chan_rate[name].extend(ch_rate)


class PcDebugOutputsLogger(DebugOutputsLogger):
    def _log_output(self, mode, key, input, output, sample_idx, context):
        if sample_idx > 4:
            return

        context = {"mode": mode, "key": key, **(context or {})}
        context_str = "_".join(f"{v}" for v in context.values())

        if (mode, key) == ("dec", "x_hat"):
            self.runner._pc_figure_logger.log(input, output, sample_idx)
            return
        else:
            arr = featuremap_image(output.cpu().numpy(), cmap=DEFAULT_COLORMAP)

        img_dir = self.runner.hparams["paths"]["images"]
        Image.fromarray(arr).save(f"{img_dir}/{sample_idx:06}_{context_str}.png")

        log_kwargs = dict(
            format="webp",
            lossless=True,
            quality=50,
            method=6,
            track_kwargs=dict(step=sample_idx),
        )
        self.runner.log_image("output", arr, context=context, **log_kwargs)


class PcFigureLogger:
    """Log point cloud figure."""

    def __init__(self, runner):
        self.runner = runner

    def log(
        self,
        x,
        x_hat,
        sample_idx,
        log_figure: bool = True,
        save_figure: bool = True,
        **kwargs,
    ):
        x = x.cpu().numpy()
        x_hat = x_hat.cpu().numpy()
        name = self.runner.hparams["model"]["name"] + "*"
        dfs = [
            point_cloud_dataframe(x, name=f"{name} (x)"),
            point_cloud_dataframe(x_hat, name=f"{name} (x_hat)"),
        ]
        df = pd.concat(dfs)
        fig = plot_point_cloud(df, **kwargs)
        if log_figure:
            context = {"name": "x_hat"}
            log_kwargs = dict(track_kwargs=dict(step=sample_idx))
            self.runner.log_figure("point-cloud", fig, context=context, **log_kwargs)
        if save_figure:
            img_dir = self.runner.hparams["paths"]["images"]
            fig.write_html(f"{img_dir}/{sample_idx:06}_x_hat.html")
        return fig


def point_cloud_dataframe(points, **kwargs):
    assert points.ndim == 2
    d = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        **{k: [v] * len(points) for k, v in kwargs.items()},
    }
    return pd.DataFrame.from_dict(d)
