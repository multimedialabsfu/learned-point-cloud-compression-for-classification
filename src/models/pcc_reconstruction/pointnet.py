from __future__ import annotations

from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers.pcc.pointnet import pointnet_g_a_simple, pointnet_g_s_simple

from .base import BaseReconstructionPccModel


@register_model("um-pcc-rec-pointnet")
class PointNetReconstructionPccModel(BaseReconstructionPccModel):
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
