from __future__ import annotations

import torch.nn as nn

from compressai.registry import register_model
from src.layers.ulhaqm.pcc import pointnet_classification_backend

from .base import BaseClassificationPcModel


@register_model("um-pc-cls-pointnet")
class PointNetClassPcModel(BaseClassificationPcModel):
    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            "g_a": [3, 64, 64, 64, 128, 1024],
            "task_backend": [1024, 512, 256, 40],
        },
    ):
        super().__init__()

        g_a_num_channels = num_channels["g_a"]

        assert num_channels["task_backend"][0] == num_channels["g_a"][-1]
        assert num_channels["task_backend"][-1] == num_classes

        self.g_a = nn.Sequential(
            *[
                x
                for ch_in, ch_out in zip(g_a_num_channels[:-1], g_a_num_channels[1:])
                for x in [
                    nn.Conv1d(ch_in, ch_out, 1),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                ]
            ],
            nn.AdaptiveMaxPool1d(1),
        )

        self.task_backend = pointnet_classification_backend(
            num_channels=num_channels["task_backend"],
        )

        self.outputs = {}
        self._setup_hooks()
