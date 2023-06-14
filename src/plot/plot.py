from __future__ import annotations

import compressai_trainer.plot.plot as _M

_PLOT_RD_SCATTER_SETTINGS_COMMON_PC = dict(
    x="bpp",
    color="name",
    hover_data=["d1-psnr", "d1-psnr-hausdorff", "epoch"],
)

_M.PLOT_RD_SETTINGS = {
    "none": {
        "scatter_kwargs": {},
        "layout_kwargs": {},
    },
    "acc_top1": {
        "scatter_kwargs": dict(
            x="bpp",
            y="acc_top1",
            color="name",
            hover_data=["acc_top1", "acc_top3", "epoch"],
        ),
        "layout_kwargs": dict(
            xaxis_title="Bit-rate [bpp]",
            xaxis=dict(range=[0.0, 2.25], tick0=0.0, dtick=0.25),
            yaxis_title="Top-1 Accuracy [%]",
            yaxis=dict(range=[0.0, 1.0], tick0=0.0, dtick=5),
        ),
    },
    "acc_top3": {
        "scatter_kwargs": dict(
            x="bpp",
            y="acc_top3",
            color="name",
            hover_data=["acc_top1", "acc_top3", "epoch"],
        ),
        "layout_kwargs": dict(
            xaxis_title="Bit-rate [bpp]",
            xaxis=dict(range=[0.0, 2.25], tick0=0.0, dtick=0.25),
            yaxis_title="Top-3 Accuracy [%]",
            yaxis=dict(range=[0.0, 1.0], tick0=0.0, dtick=5),
        ),
    },
    "d1-psnr": {
        "scatter_kwargs": dict(
            **_PLOT_RD_SCATTER_SETTINGS_COMMON_PC,
            y="d1-psnr",
        ),
        "layout_kwargs": dict(
            **_M._PLOT_RD_LAYOUT_SETTINGS_COMMON,
            yaxis_title="D1 PSNR [dB]",
            yaxis=dict(dtick=1),
            # yaxis=dict(range=[20, 50], tick0=20, dtick=1),
        ),
    },
    "d1-psnr-hausdorff": {
        "scatter_kwargs": dict(
            **_PLOT_RD_SCATTER_SETTINGS_COMMON_PC,
            y="d1-psnr-hausdorff",
        ),
        "layout_kwargs": dict(
            **_M._PLOT_RD_LAYOUT_SETTINGS_COMMON,
            yaxis_title="Hausdorff D1 PSNR [dB]",
            yaxis=dict(dtick=1),
            # yaxis=dict(range=[20, 50], tick0=20, dtick=1),
        ),
    },
}
