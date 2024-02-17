from __future__ import annotations

import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from compressai.registry import register_criterion

from .utils import compute_rate_loss


@register_criterion("ChamferPccRateDistortionLoss")
class ChamferPccRateDistortionLoss(nn.Module):
    """Simple loss for regular point cloud compression.

    For compression models that reconstruct the input point cloud.
    """

    LMBDA_DEFAULT = {
        # "bpp": 1.0,
        "rec": 1.0,
    }

    def __init__(self, lmbda=None, rate_key="bpp"):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.lmbda.setdefault(rate_key, 1.0)

    def forward(self, output, target):
        out = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
        }

        out["loss"] = sum(
            self.lmbda[k] * out[f"{k}_loss"]
            for k in self.lmbda.keys()
            if f"{k}_loss" in out
        )

        return out

    def compute_rate_loss(self, output, target):
        if "likelihoods" not in output:
            return {}
        N, P, _ = target["pos"].shape
        return compute_rate_loss(output["likelihoods"], N, P)

    def compute_rec_loss(self, output, target):
        loss_chamfer, _ = chamfer_distance(output["x_hat"], target["pos"])
        return {"rec_loss": loss_chamfer}


@register_criterion("MultitaskPccRateDistortionLoss")
class MultitaskPccRateDistortionLoss(nn.Module):
    """
    Abbreviations:
    - rec = reconstruction
    - cls = classification
    - fm  = feature-matching
    """

    LMBDA_DEFAULT = {
        # "bpp": 1.0,
        "rec": 1.0,
        "cls": 1.0,
    }

    def __init__(self, lmbda=None, rate_key="bpp", target_label_key="label"):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.lmbda.setdefault(rate_key, 1.0)
        self.rec_metric = lambda *args, **kwargs: chamfer_distance(*args, **kwargs)[0]
        self.cls_metric = nn.CrossEntropyLoss()
        self.fm_metric = nn.MSELoss()
        self.target_label_key = target_label_key

    def forward(self, output, target):
        out = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
            **self.compute_cls_loss(output, target),
            **self.compute_fm_loss(output, target),
        }

        out["loss"] = sum(
            self.lmbda[k] * out[f"{k}_loss"]
            for k in self.lmbda.keys()
            if f"{k}_loss" in out
        )

        return out

    def compute_rate_loss(self, output, target):
        if "likelihoods" not in output:
            return {}
        N, P, _ = target["pos"].shape
        return compute_rate_loss(output["likelihoods"], N, P)

    def compute_rec_loss(self, output, target):
        if "x_hat" not in output:
            return {}
        return {"rec_loss": self.rec_metric(output["x_hat"], target["pos"])}

    def compute_cls_loss(self, output, target):
        if "t_hat" not in output:
            return {}
        return {
            "cls_loss": self.cls_metric(
                output["t_hat"],
                target[self.target_label_key].squeeze(1),
            )
        }

    def compute_fm_loss(self, output, target):
        return {}
