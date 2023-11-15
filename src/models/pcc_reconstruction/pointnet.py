from __future__ import annotations

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.registry import register_model
from src.layers.ulhaqm.pcc import pointnet_g_a_simple, pointnet_g_s_simple

from .base import BasePccModel


@register_model("um-pcc-rec-pointnet")
class PointnetPccModel(BasePccModel):
    def __init__(
        self,
        num_points=1024,
        num_channels={
            "g_a": {
                "pointwise": [3, 64, 64, 64, 128, 1024],
            },
            "g_s": {
                "pointwise": [1024, 256, 512, 1024 * 3],
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

        num_channels_g_s = [
            *num_channels["g_s"]["pointwise"],
        ]

        assert num_channels_g_a[-1] == num_channels_g_s[0]
        assert num_channels_g_s[-1] == num_points * 3

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups.get("g_a"))

        self.g_s = pointnet_g_s_simple(num_channels["g_s"]["pointwise"])

        self.latent_codec = EntropyBottleneckLatentCodec(
            N=num_channels_g_a[-1],
            entropy_bottleneck=EntropyBottleneck(
                num_channels_g_a[-1],
                tail_mass=1e-4,
            ),
        )
