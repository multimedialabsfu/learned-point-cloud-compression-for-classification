from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.ulhaqm import NamedLayer
from src.layers.ulhaqm.pcc import (
    pointnet_classification_backend,
    pointnet_g_a_simple,
    pointnet_g_s_simple,
)


class BaseMultitaskPccModel(CompressionModel):
    latent_codec: Mapping[str, LatentCodec]

    def _setup_hooks(self):
        def hook(module, input, output):
            self.outputs[module.name] = output

        for _, module in self.task_backend.named_modules():
            if not isinstance(module, NamedLayer):
                continue
            module.register_forward_hook(hook)

    def forward(self, input):
        self.outputs = {}
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y1, y2 = self.split(y)
        y1_out = self.latent_codec["y1"](y1)
        if y2.shape[1] != 0:
            y2_out = self.latent_codec["y2"](y2)
        else:
            y2_out = {
                "y_hat": torch.zeros_like(y2),
                "likelihoods": {"y": torch.ones_like(y2)},
            }
        y1_hat = y1_out["y_hat"]
        y2_hat = y2_out["y_hat"]
        t_hat = self.task_backend(y1_hat)
        y_hat = self.merge(y1_hat, y2_hat)
        y1_hat_rec = y1_hat.detach() if self.detach_y1_hat else y1_hat
        y_hat_rec = self.merge(y1_hat_rec, y2_hat)
        x_hat = self.g_s(y_hat_rec)
        assert x_hat.shape == x.shape

        return {
            "x_hat": x_hat,
            "t_hat": t_hat,
            "likelihoods": {
                "y1": y1_out["likelihoods"]["y"],
                "y2": y2_out["likelihoods"]["y"],
            },
            **{k: v for k, v in self.outputs.items()},
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        self.outputs = {}
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y1, y2 = self.split(y)
        y1_out = self.latent_codec["y1"].compress(y1)
        [y1_strings] = y1_out["strings"]
        if y2.shape[1] != 0:
            y2_out = self.latent_codec["y2"].compress(y2)
            [y2_strings] = y2_out["strings"]
        else:
            y2_strings = b""
        # NOTE: the shape in y*_out["shape"] is incorrectly y.shape[-2:]
        return {"strings": [y1_strings, y2_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        self.outputs = {}
        [y1_strings, y2_strings] = strings
        y1_out = self.latent_codec["y1"].decompress([y1_strings], shape)
        if y2_strings != b"":
            y2_out = self.latent_codec["y2"].decompress([y2_strings], shape)
        else:
            shape = list(y1_out["y_hat"].shape)
            shape[1] = 0
            y2_out = {"y_hat": torch.empty(shape, device=y1_out["y_hat"].device)}
        y1_hat = y1_out["y_hat"]
        y2_hat = y2_out["y_hat"]
        t_hat = self.task_backend(y1_hat)
        y_hat = self.merge(y1_hat, y2_hat)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "t_hat": t_hat}

    def split(self, y):
        y1 = y[:, : self.num_split_channels[0]]
        y2 = y[:, self.num_split_channels[0] :]
        return y1, y2

    def merge(self, *args):
        return torch.cat(args, dim=1)


@register_model("um-pcc-multitask-cls-pointnet")
class PointNetClassMultitaskPccModel(BaseMultitaskPccModel):
    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            "g_a": {
                "pointwise": [3, 64, 64, 64, 128, 1024],
                "mixer": [],
            },
            "g_s": {
                "pointwise": [1024, 256, 512, 1024 * 3],
            },
            "task_backend": {
                "transform": {
                    "pointwise": [],
                    "mixer": [],
                },
                "mlp": [1024, 512, 256, 40],
            },
        },
        groups={
            "g_a": {
                "pointwise": [1, 1, 1, 1, 1],
            },
            "task_backend": {
                "transform": {
                    "pointwise": [],
                },
            },
        },
        detach_y1_hat=True,
    ):
        super().__init__()

        self.detach_y1_hat = detach_y1_hat

        num_channels_g_a = [
            *num_channels["g_a"]["pointwise"],
            *num_channels["g_a"]["mixer"][1:],
        ]

        num_channels_task_backend = [
            *num_channels["task_backend"]["transform"]["pointwise"],
            *num_channels["task_backend"]["transform"]["mixer"][1:],
            *num_channels["task_backend"]["mlp"][1:],
        ]

        num_channels_g_s = [
            *num_channels["g_s"]["pointwise"],
        ]

        self.num_split_channels = [
            num_channels_task_backend[0],
            num_channels_g_a[-1] - num_channels_task_backend[0],
        ]

        assert num_channels_task_backend[-1] == num_classes
        assert num_channels_g_a[-1] == num_channels_g_s[0]
        assert num_channels_g_s[-1] == num_points * 3

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups.get("g_a"))

        self.g_s = pointnet_g_s_simple(num_channels["g_s"]["pointwise"])

        self.task_backend = nn.Sequential(
            *pointnet_g_a_simple(
                num_channels["task_backend"]["transform"],
                groups.get("task_backend", {}).get("transform"),
                gain=1.0,
            ),
            *pointnet_classification_backend(
                num_channels=num_channels["task_backend"]["mlp"],
            ),
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y1": EntropyBottleneckLatentCodec(
                    N=self.num_split_channels[0],
                    entropy_bottleneck=EntropyBottleneck(
                        self.num_split_channels[0], tail_mass=1e-4
                    ),
                ),
                "y2": EntropyBottleneckLatentCodec(
                    N=self.num_split_channels[1],
                    entropy_bottleneck=EntropyBottleneck(
                        self.num_split_channels[1], tail_mass=1e-4
                    ),
                ),
            }
        )

        # Allow empty y2 channels.
        if self.num_split_channels[1] == 0:
            self.latent_codec.pop("y2")

        self.outputs = {}
        self._setup_hooks()
