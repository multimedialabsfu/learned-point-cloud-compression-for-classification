from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import torch
from catalyst import metrics

from compressai.models.base import CompressionModel
from compressai.typing import TCriterion
from compressai_trainer.registry import register_runner
from compressai_trainer.runners.base import BaseRunner
from compressai_trainer.utils.metrics import compute_metrics

from .utils import (
    EbDistributionsFigureLogger,
    GradientClipper,
    PcChannelwiseBppMeter,
    PcDebugOutputsLogger,
    PcFigureLogger,
    RdFigureLogger,
)

RD_PLOT_METRICS = [
    "rec_loss",
    # "d1-psnr",
    # "d1-psnr-hausdorff",
    # "d2-psnr",
    # "d2-psnr-hausdorff",
]

RD_PLOT_DESCRIPTIONS = [
    "Reconstruction loss",
    # "D1-PSNR (point-to-point)",
    # "Hausdorff D1-PSNR (point-to-point)",
    # "D2-PSNR (point-to-plane)",
    # "Hausdorff D2-PSNR (point-to-plane)",
]

RD_PLOT_TITLE = "Performance evaluation on {dataset} - {metric}"

RD_PLOT_SETTINGS_COMMON: dict[str, Any] = dict(
    codecs=[
        #
    ],
    scatter_kwargs=dict(
        hover_data=[
            "name",
            "bpp",
            # "d1-psnr",
            # "d1-psnr-hausdorff",
            # "d2-psnr",
            # "d2-psnr-hausdorff",
            "loss",
            "rec_loss",
            "epoch",
            "criterion.lmbda",
        ],
    ),
)


@register_runner("ReconstructionPointCloudCompressionRunner")
@register_runner("PointCloudCompressionRunner")  # NOTE: Old name.
class ReconstructionPointCloudCompressionRunner(BaseRunner):
    """Runner for point-cloud compression experiments."""

    def __init__(
        self,
        inference: dict[str, Any],
        meters: dict[str, list[str]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._inference_kwargs = inference
        self._meters = meters
        self._grad_clip = GradientClipper(self)
        self._debug_outputs_logger = PcDebugOutputsLogger(self)
        self._eb_distributions_figure_logger = EbDistributionsFigureLogger(self)
        self._pc_figure_logger = PcFigureLogger(self)
        self._rd_figure_logger = RdFigureLogger(self)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self._setup_loader_metrics()
        self._setup_meters()

    def handle_batch(self, batch):
        if self.is_infer_loader:
            return self._handle_batch_infer(batch)

        out_net = self.model(batch)
        out_criterion = self.criterion(out_net, batch)
        loss = {}
        loss["net"] = out_criterion["loss"]

        if self.is_train_loader:
            loss["net"].backward()
            self._grad_clip()
            self.optimizer["net"].step()

        loss["aux"] = self.model_module.aux_loss()

        if self.is_train_loader:
            loss["aux"].backward()
            self.optimizer["aux"].step()
            self.optimizer["net"].zero_grad()
            self.optimizer["aux"].zero_grad()

        batch_metrics = {
            "loss": loss["net"],
            "aux_loss": loss["aux"],
            **out_criterion,
        }
        self._update_batch_metrics(batch_metrics)

    def _handle_batch_infer(self, batch):
        x = batch["points"]
        out_infer = self.predict_batch(batch, **self._inference_kwargs)
        out_net = out_infer["out_net"]

        out_criterion = self.criterion(out_net, batch)
        # out_metrics = compute_metrics(out_net, batch, ["pc_error"])
        out_metrics = compute_metrics(out_net, batch, [])
        out_metrics["bpp"] = out_infer["bpp"]
        out_metrics["bpp_loss"] = out_criterion["bpp_loss"].item()
        out_metrics["rec_loss"] = out_criterion["rec_loss"].item()

        loss = {}
        loss["net"] = out_criterion["loss"]
        loss["aux"] = self.model_module.aux_loss()

        batch_metrics = {
            "loss": loss["net"],
            "aux_loss": loss["aux"],
            **out_criterion,
            **out_metrics,
            "bpp": out_infer["bpp"],
        }
        self._update_batch_metrics(batch_metrics)
        self._handle_custom_metrics(out_net, out_metrics, batch)

        self._debug_outputs_logger.log(x, out_infer)

    def predict_batch(self, batch, **kwargs):
        batch = {k: v.to(self.engine.device) for k, v in batch.items()}
        return inference(self.model_module, batch, criterion=self.criterion, **kwargs)

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.is_infer_loader:
            self._log_rd_curves()
            self._log_eb_distributions()
            self._loader_metrics["chan_bpp"].log()

    @property
    def _current_dataframe(self):
        r = lambda x: float(f"{x:.4g}")
        d = {
            "name": self.hparams["model"]["name"] + "*",
            "epoch": self.epoch_step,
            "criterion.lmbda": self.hparams["criterion"]["lmbda"],
            "loss": r(self.loader_metrics["loss"]),
            "bpp": r(self.loader_metrics["bpp"]),
            "bpp_loss": r(self.loader_metrics["bpp_loss"]),
            "rec_loss": r(self.loader_metrics["rec_loss"]),
            # "d1-psnr": r(self.loader_metrics["d1-psnr"]),
            # "d1-psnr-hausdorff": r(self.loader_metrics["d1-psnr-hausdorff"]),
        }
        return pd.DataFrame.from_dict([d])

    def _current_rd_traces(self, metric):
        lmbda = self.hparams["criterion"]["lmbda"]
        num_points = len(self._loader_metrics["bpp"])
        samples_scatter = go.Scatter(
            x=self._loader_metrics["bpp"],
            y=self._loader_metrics[metric],
            mode="markers",
            name=f'{self.hparams["model"]["name"]} {lmbda:.4f}',
            text=[f"lmbda={lmbda:.4f}\nsample_idx={i}" for i in range(num_points)],
            visible="legendonly",
        )
        return [samples_scatter]

    def _handle_custom_metrics(self, out_net, out_metrics, input):
        self._loader_metrics["chan_bpp"].update(out_net, input)
        for metric in ["bpp", *RD_PLOT_METRICS]:
            self._loader_metrics[metric].append(out_metrics[metric])

    def _log_eb_distributions(self):
        self._eb_distributions_figure_logger.log(
            log_kwargs=dict(track_kwargs=dict(step=0))
        )

    def _log_rd_curves(self):
        meta = self.hparams["dataset"]["infer"]["meta"]
        for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
            self._rd_figure_logger.log(
                df=self._current_dataframe,
                traces=self._current_rd_traces(metric),
                metric=metric,
                dataset=meta["identifier"],
                **RD_PLOT_SETTINGS_COMMON,
                layout_kwargs=dict(
                    title=RD_PLOT_TITLE.format(
                        dataset=meta["name"],
                        metric=description,
                    ),
                ),
            )

    def _setup_loader_metrics(self):
        self._loader_metrics = {
            "chan_bpp": PcChannelwiseBppMeter(self),
            **{k: [] for k in ["bpp", *RD_PLOT_METRICS]},
        }

    def _setup_meters(self):
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in self._meters[self.loader_key]
        }


@torch.no_grad()
def inference(
    model: CompressionModel,
    input: dict[str, torch.Tensor],
    skip_compress: bool = False,
    skip_decompress: bool = False,
    criterion: Optional[TCriterion] = None,
) -> dict[str, Any]:
    """Run compression model on image batch."""
    x = input["points"]
    N, P, C = x.shape
    assert C == 3

    # Compress using forward.
    out_net = model(input)

    # Compress using compress/decompress.
    if not skip_compress:
        start = time.time()
        out_enc = model.compress(input)
        enc_time = time.time() - start
    else:
        out_enc = {}
        enc_time = None

    if not skip_decompress:
        assert not skip_compress
        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start
    else:
        out_dec = dict(out_net)
        del out_dec["likelihoods"]
        dec_time = None

    # Compute bpp.
    if not skip_compress:
        num_bits = sum(sum(map(len, s)) for s in out_enc["strings"]) * 8.0
        num_points = N * P
        bpp = num_bits / num_points
    else:
        out_criterion = criterion(out_net, input)
        bpp = out_criterion["bpp_loss"].item()

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }
