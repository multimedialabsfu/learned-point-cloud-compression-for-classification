from __future__ import annotations

import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.ulhaqm import Gain
from src.layers.ulhaqm.layers import Lambda
from src.layers.ulhaqm.pcc import (
    GAIN,
    conv1d_group_seq,
    pointnet_g_a_simple,
    pointnet_g_s_simple,
)


class BasePccModel(CompressionModel):
    latent_codec: LatentCodec

    def forward(self, input):
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        assert x_hat.shape == x.shape

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec.compress(y)
        [y_strings] = y_out["strings"]
        return {"strings": [y_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        [y_strings] = strings
        y_hat = self.latent_codec.decompress([y_strings], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}


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
