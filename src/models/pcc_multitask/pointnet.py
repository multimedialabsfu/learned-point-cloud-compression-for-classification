from __future__ import annotations

from typing import Mapping

import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers import Gain
from src.layers.pcc import (
    pointnet_classification_backend,
    pointnet_g_a_simple,
    pointnet_g_s_simple,
)

from .base import BaseMultitaskPccModel


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
            },
            "g_s": {
                "pointwise": [1024, 256, 512, 1024 * 3],
            },
            "task_backend": {
                "mlp": [1024, 512, 256, 40],
            },
        },
        groups={
            "g_a": {
                "pointwise": [1, 1, 1, 1, 1],
            },
            "task_backend": {},
        },
        detach_y1_hat=True,
    ):
        super().__init__()

        self.detach_y1_hat = detach_y1_hat

        num_channels_g_a = [
            *num_channels["g_a"]["pointwise"],
        ]

        num_channels_task_backend = [
            *num_channels["task_backend"]["mlp"],
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
            *nn.Sequential(
                nn.Identity(),  # For compatibility with previous checkpoints.
                Gain((num_channels_task_backend[0], 1), factor=1.0),
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
