from __future__ import annotations

from typing import Mapping

import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers import Gain
from src.layers.pcc import GAIN, pointnet_classification_backend, pointnet_g_a_simple

from .base import BaseClassificationPccModel


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
                "mlp": [1024, 512, 256, 40],
            },
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
        num_channels_task_backend = [
            *num_channels["task_backend"]["mlp"],
        ]

        assert num_channels_task_backend[0] == num_channels_g_a[-1]
        assert num_channels_task_backend[-1] == num_classes

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.task_backend = nn.Sequential(
            nn.Sequential(
                nn.Identity(),  # For compatibility with previous checkpoints.
                Gain((num_channels_task_backend[0], 1), GAIN),
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
