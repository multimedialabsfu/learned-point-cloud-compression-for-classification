from __future__ import annotations

from typing import Mapping

import torch.nn as nn

from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers import Gain
from src.layers.pcc.pointnet import (
    GAIN,
    pointnet_classification_backend,
    pointnet_g_a_simple,
)

from .base import BaseClassificationPccModel


@register_model("um-pcc-cls-only-pointnet")
@register_model("um-pcc-cls-only-pointnet-mmsp2023")
class PointNetClassOnlyPccModel(BaseClassificationPccModel):
    """PointNet-based PCC classification model.

    Model based on PointNet [Qi2017]_, modified for compression for
    reconstruction by [Yan2019]_, and applied to compression for
    classification in [Ulhaq2023]_.

    References:

        .. [Qi2017] `"PointNet: Deep Learning on Point Sets for
            3D Classification and Segmentation"
            <https://arxiv.org/abs/1612.00593>`_, by Charles R. Qi,
            Hao Su, Kaichun Mo, and Leonidas J. Guibas, CVPR 2017.

        .. [Yan2019] `"Deep AutoEncoder-based Lossy Geometry Compression
            for Point Clouds" <https://arxiv.org/abs/1905.03691>`_,
            by Wei Yan, Yiting Shao, Shan Liu, Thomas H Li, Zhu Li,
            and Ge Li, 2019.

        .. [Ulhaq2023] `"Learned Point Cloud Compression for
            Classification" <https://arxiv.org/abs/2308.05959>`_,
            by Mateen Ulhaq and Ivan V. Bajić, MMSP 2023.
    """

    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            "g_a": [3, 64, 64, 64, 128, 1024],
            "task_backend": [1024, 512, 256, 40],
        },
        groups={
            "g_a": [1, 1, 1, 1, 1],
        },
    ):
        super().__init__()

        assert num_channels["task_backend"][0] == num_channels["g_a"][-1]
        assert num_channels["task_backend"][-1] == num_classes

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.task_backend = nn.Sequential(
            Gain((num_channels["task_backend"][0], 1), GAIN),
            *pointnet_classification_backend(num_channels["task_backend"]),
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y": EntropyBottleneckLatentCodec(
                    channels=num_channels["g_a"][-1],
                    tail_mass=1e-4,
                ),
            }
        )

        self.outputs = {}
        self._setup_hooks()
