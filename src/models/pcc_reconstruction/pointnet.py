from __future__ import annotations

from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers.pcc.pointnet import pointnet_g_a_simple, pointnet_g_s_simple

from .base import BaseReconstructionPccModel


@register_model("um-pcc-rec-pointnet")
class PointNetReconstructionPccModel(BaseReconstructionPccModel):
    """PointNet-based PCC reconstruction model.

    Model based on PointNet [Qi2017]_, modified for compression by
    [Yan2019]_, with layer configurations and other modifications as
    used in [Ulhaq2023]_.

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

    def __init__(
        self,
        num_points=1024,
        num_channels={
            "g_a": [3, 64, 64, 64, 128, 1024],
            "g_s": [1024, 256, 512, 1024 * 3],
        },
        groups={
            "g_a": [1, 1, 1, 1, 1],
        },
    ):
        super().__init__()

        assert num_channels["g_a"][-1] == num_channels["g_s"][0]
        assert num_channels["g_s"][-1] == num_points * 3

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.g_s = pointnet_g_s_simple(num_channels["g_s"])

        self.latent_codec = EntropyBottleneckLatentCodec(
            channels=num_channels["g_a"][-1],
            tail_mass=1e-4,
        )
