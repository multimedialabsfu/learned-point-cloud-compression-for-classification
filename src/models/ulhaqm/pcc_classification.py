from __future__ import annotations

from typing import Mapping

import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.ulhaqm import NamedLayer
from src.layers.ulhaqm.pcc import pointnet_classification_backend, pointnet_g_a_simple


class BaseClassificationPccModel(CompressionModel):
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
        y_out = self.latent_codec["y"](y)
        y_hat = y_out["y_hat"]
        t_hat = self.task_backend(y_hat)

        return {
            "t_hat": t_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
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
        y_out = self.latent_codec["y"].compress(y)
        [y_strings] = y_out["strings"]
        # NOTE: the shape in y_out["shape"] is incorrectly y.shape[-2:]
        return {"strings": [y_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        self.outputs = {}
        [y_strings] = strings
        y_out = self.latent_codec["y"].decompress([y_strings], shape)
        y_hat = y_out["y_hat"]
        t_hat = self.task_backend(y_hat)
        return {"t_hat": t_hat}


@register_model("um-pcc-cls-only-pointnet")
class PointNetClassOnlyPccModel(BaseClassificationPccModel):
    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            "g_a": {
                "pointwise": [3, 64, 64, 64, 128, 1024],
            },
            "task_backend": [1024, 512, 256, 40],
        },
        groups={
            "g_a": {
                "pointwise": [1, 1, 1, 1, 1],
            },
        },
    ):
        super().__init__()

        num_channels_g_a = [
            *num_channels["g_a"]["pointwise"],
        ]

        assert num_channels["task_backend"][0] == num_channels_g_a[-1]
        assert num_channels["task_backend"][-1] == num_classes

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.task_backend = pointnet_classification_backend(
            num_channels=num_channels["task_backend"],
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y": EntropyBottleneckLatentCodec(
                    N=num_channels_g_a[-1],
                    entropy_bottleneck=EntropyBottleneck(
                        num_channels_g_a[-1], tail_mass=1e-4
                    ),
                ),
            }
        )

        self.outputs = {}
        self._setup_hooks()


@register_model("um-pcc-cls-only-pointnet-mmsp2023")
@register_model("um-pcc-cls-only-pointnet-mini-001")  # NOTE: Old name.
class PointNetClassOnlyPccModelMmsp2023(BaseClassificationPccModel):
    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            "g_a": {
                "pointwise": [3, 64, 64, 64, 128, 1024],
            },
            "task_backend": {
                "transform": {
                    "pointwise": [1024],
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
    ):
        super().__init__()

        num_channels_g_a = [
            *num_channels["g_a"]["pointwise"],
        ]
        num_channels_task_backend = [
            *num_channels["task_backend"]["transform"]["pointwise"],
            *num_channels["task_backend"]["mlp"][1:],
        ]

        assert num_channels_task_backend[0] == num_channels_g_a[-1]
        assert num_channels_task_backend[-1] == num_classes

        # FIXME: Disabled since this hasn't been implemented correctly yet.
        # Should probably be implemented in a separate model, to avoid confusion.
        assert len(num_channels["task_backend"]["transform"]["pointwise"]) == 1

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.task_backend = nn.Sequential(
            pointnet_g_a_simple(
                num_channels["task_backend"]["transform"],
                groups["task_backend"]["transform"],
            ),
            pointnet_classification_backend(
                num_channels=num_channels["task_backend"]["mlp"],
            ),
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y": EntropyBottleneckLatentCodec(
                    N=num_channels_g_a[-1],
                    entropy_bottleneck=EntropyBottleneck(
                        num_channels_g_a[-1], tail_mass=1e-4
                    ),
                ),
            }
        )

        self.outputs = {}
        self._setup_hooks()
