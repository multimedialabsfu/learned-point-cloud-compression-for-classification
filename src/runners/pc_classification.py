from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from catalyst import metrics

from compressai_trainer.registry import register_runner
from compressai_trainer.runners.base import BaseRunner
from src.utils.metrics import compute_metrics

from .utils import GradientClipper, PcDebugOutputsLogger, PcFigureLogger

RD_PLOT_METRICS = [
    "acc_top1",
    "acc_top3",
]


@register_runner("PointCloudClassificationRunner")
class PointCloudClassificationRunner(BaseRunner):
    """Runner for single-task point-cloud classification experiments."""

    def __init__(
        self,
        inference: dict[str, Any],
        meters: dict[str, list[str]],
        metrics: list[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._inference_kwargs = inference
        self._meters = meters
        self._metrics = metrics
        self._grad_clip = GradientClipper(self)
        self._debug_outputs_logger = PcDebugOutputsLogger(self)
        self._pc_figure_logger = PcFigureLogger(self)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
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
            self.optimizer["net"].zero_grad()

        batch_metrics = {
            "loss": loss["net"],
            **out_criterion,
        }
        self._update_batch_metrics(batch_metrics)

    def _handle_batch_infer(self, batch):
        x = batch["points"]
        out_infer = self.predict_batch(batch, **self._inference_kwargs)
        out_net = out_infer["out_net"]

        out_criterion = self.criterion(out_net, batch)
        out_metrics = compute_metrics(batch, out_net, self._metrics)

        loss = {}
        loss["net"] = out_criterion["loss"]

        batch_metrics = {
            "loss": loss["net"],
            **out_criterion,
            **out_metrics,
        }
        self._update_batch_metrics(batch_metrics)

        self._debug_outputs_logger.log(x, out_infer)

    def predict_batch(self, batch, **kwargs):
        batch = {k: v.to(self.engine.device) for k, v in batch.items()}
        return inference(self.model_module, batch, **kwargs)

    def on_loader_end(self, runner):
        super().on_loader_end(runner)

    @property
    def _current_dataframe(self):
        r = lambda x: float(f"{x:.4g}")  # noqa: E731
        d = {
            "name": self.hparams["model"]["name"] + "*",
            "epoch": self.epoch_step,
            "loss": r(self.loader_metrics["loss"]),
            "acc_top1": r(self.loader_metrics["acc_top1"]),
            "acc_top3": r(self.loader_metrics["acc_top3"]),
        }
        return pd.DataFrame.from_dict([d])

    def _setup_meters(self):
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in self._meters[self.loader_key]
        }


@torch.no_grad()
def inference(
    model: nn.Module,
    input: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Run model on image batch."""
    x = input["points"]
    N, P, C = x.shape
    assert C == 3

    out_net = model(input)

    return {
        "out_net": out_net,
    }
