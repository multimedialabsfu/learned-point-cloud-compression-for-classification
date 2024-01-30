from __future__ import annotations

import torch.nn as nn

from compressai.registry import register_model
from src.layers.pcc.pointnet import conv1d_group_seq, pointnet_classification_backend

from .base import BaseClassificationPcModel


@register_model("um-pc-cls-pointnet")
class PointNetClassPcModel(BaseClassificationPcModel):
    """PointNet classification model.

    Model by [Qi2017]_.

    .. note::

        This simplified implementation does not use the "input
        transform" and "feature transform" layers from the original
        paper. In Table 5, the authors report that these layers may
        improve the classification accuracy by 2.1%.

    References:

        .. [Qi2017] `"PointNet: Deep Learning on Point Sets for
            3D Classification and Segmentation"
            <https://arxiv.org/abs/1612.00593>`_, by Charles R. Qi,
            Hao Su, Kaichun Mo, and Leonidas J. Guibas, CVPR 2017.
    """

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

        assert num_channels["task_backend"][0] == num_channels["g_a"][-1]
        assert num_channels["task_backend"][-1] == num_classes

        self.g_a = nn.Sequential(
            *conv1d_group_seq(num_channels["g_a"]),
            nn.AdaptiveMaxPool1d(1),
        )

        self.task_backend = pointnet_classification_backend(
            num_channels=num_channels["task_backend"],
        )

        self.outputs = {}
        self._setup_hooks()
