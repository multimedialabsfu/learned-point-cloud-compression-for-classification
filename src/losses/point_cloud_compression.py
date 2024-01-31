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

    def __init__(self, lmbda=None, rate_key="bpp"):
        super().__init__()
        self.lmbda = lmbda if isinstance(lmbda, dict) else {"rec": lmbda}
        self.rate_key = rate_key

    def forward(self, output, target):
        out = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
        }
        out["loss"] = out[f"{self.rate_key}_loss"] + self.lmbda["rec"] * out["rec_loss"]
        return out

    def compute_rate_loss(self, output, target):
        N, P, C = target["pos"].shape
        assert C == 3
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

    Example config:

    .. code-block:: python

        lmbda = {
            "rec": 1.0,
            "cls": 1.0,
            "fm": {
                "1": 1.0,
                "2": 1.0,
            },
        }
    """

    def __init__(self, lmbda={}, rate_key="bpp"):
        super().__init__()
        self.lmbda = lmbda
        self.rate_key = rate_key
        self.rec_metric = lambda *args, **kwargs: chamfer_distance(*args, **kwargs)[0]
        self.cls_metric = nn.CrossEntropyLoss()
        self.fm_metric = nn.MSELoss()

    def forward(self, output, target):
        out = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
            **self.compute_cls_loss(output, target),
            **self.compute_fm_loss(output, target),
        }

        out["loss"] = (
            out.get(f"{self.rate_key}_loss", 0)
            + self.lmbda.get("rec", 0) * out.get("rec_loss", 0)
            + self.lmbda.get("cls", 0) * out.get("cls_loss", 0)
            # + sum(lmbda * out[f"fm_{k}_loss"] for k, lmbda in self.lmbda["fm"].items())
        )

        return out

    def compute_rate_loss(self, output, target):
        if "likelihoods" not in output:
            return {}
        N, P, C = target["pos"].shape
        assert C == 3
        return compute_rate_loss(output["likelihoods"], N, P)

    def compute_rec_loss(self, output, target):
        if "x_hat" not in output:
            return {}
        return {"rec_loss": self.rec_metric(output["x_hat"], target["pos"])}
        # return {"rec_loss": self.rec_metric(output["x_hat"], target["x"])}

    def compute_cls_loss(self, output, target):
        if "t_hat" not in output:
            return {}
        return {
            "cls_loss": self.cls_metric(output["t_hat"], target["label"].squeeze(1))
        }
        # return {"cls_loss": self.cls_metric(output["t_hat"], target["t"])}

    def compute_fm_loss(self, output, target):
        return {}
        return {
            f"fm_{k}_loss": self.fm_metric(output[f"s_{k}_hat"], output[f"s_{k}"])
            # f"fm_{k}_loss": self.fm_metric(output[f"s_{k}_hat"], target[f"s_{k}"])
            for k in self.lmbda["fm"]
        }
