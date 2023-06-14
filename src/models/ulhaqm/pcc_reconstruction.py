from __future__ import annotations

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.ulhaqm import Gain, Reshape, Transpose, conv1d
from src.layers.ulhaqm.pcc import GAIN

# How is this supposed to reduce spatial correlations...?!
# 1x1 convs means each point is individual...?!

# normalize inputs pre-model()
# reshape entropy_bottleneck inputs to 3 * c?
# reshape outputs to 3 * ...?

# Compare with pccAI model (full MLP)

# x.shape=torch.Size([32, 2048, 3])
# x_t.shape=torch.Size([32, 3, 2048])
# y.shape=torch.Size([32, 512, 1])
# y_hat.shape=torch.Size([32, 512, 1])
# x_hat.shape=torch.Size([32, 2048, 3])

# points (N, 3)
# voxels (H, W)
# octree (?, 8)


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


@register_model("um-pcc-example")
class ExamplePccModel(BasePccModel):
    def __init__(self, num_points=2048, num_bottleneck_channels=512):
        super().__init__()

        g_a_num_channels = [3, 64, 128, 128, 256, num_bottleneck_channels]
        g_s_num_channels = [num_bottleneck_channels, 256, 256, num_points * 3]

        # TODO maybe the final Conv-ReLU-BatchNorm meant we are max-pooling [-3, 3]...
        # and the quantized result is too low range?
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

        # TODO Conv1d may not be equivalent to Linear if not channelwise...
        # TODO Try Reshape((g_s_num_channels[0],)) and then Linear just in case...
        # Also, Pointnet2_PyTorch uses bias=False
        self.g_s = nn.Sequential(
            *[
                x
                for ch_in, ch_out in zip(g_s_num_channels[:-2], g_s_num_channels[1:-1])
                for x in [
                    nn.Conv1d(ch_in, ch_out, 1),
                    nn.ReLU(inplace=True),
                ]
            ],
            nn.Conv1d(g_s_num_channels[-2], g_s_num_channels[-1], 1),
            Reshape((num_points, 3)),
        )

        self.latent_codec = EntropyBottleneckLatentCodec(
            N=g_a_num_channels[-1],
            entropy_bottleneck=EntropyBottleneck(
                g_a_num_channels[-1],
                tail_mass=1e-4,
            ),
        )


@register_model("um-pcc-simple")
class SimplePccModel(BasePccModel):
    def __init__(self, num_points=2048, num_bottleneck_channels=2048 * 3, gain=10):
        super().__init__()

        assert num_points * 3 % num_bottleneck_channels == 0

        class Analysis(nn.Module):
            def __init__(self, num_points, num_bottleneck_channels, gain):
                super().__init__()
                self.num_points = num_points
                self.num_bottleneck_channels = num_bottleneck_channels
                self.gain = gain

            def forward(self, x):
                n, c, p = x.shape
                assert c == 3 and p == self.num_points
                x = x.reshape(
                    n,
                    self.num_bottleneck_channels,
                    c * p // self.num_bottleneck_channels,
                )
                x = x * self.gain
                return x

        class Synthesis(nn.Module):
            def __init__(self, num_points, num_bottleneck_channels, gain):
                super().__init__()
                self.num_points = num_points
                self.num_bottleneck_channels = num_bottleneck_channels
                self.gain = gain
                self.param = nn.Parameter(torch.ones(1))

            def forward(self, x):
                n, _, _ = x.shape
                x = x / self.gain * self.param
                x = x.reshape(n, 3, self.num_points)
                x = x.transpose(-2, -1)
                return x

        # self.g_a = Analysis(num_points, num_bottleneck_channels, gain)
        # self.g_s = Synthesis(num_points, num_bottleneck_channels, gain)

        self.g_a = nn.Sequential(
            Reshape(
                (num_bottleneck_channels, 3 * num_points // num_bottleneck_channels)
            ),
            Gain((1,), gain),
        )

        self.g_s = nn.Sequential(
            Gain((1,), 1 / gain),
            Reshape((3, num_points)),
            Transpose(-2, -1),
        )

        self.latent_codec = EntropyBottleneckLatentCodec(
            N=num_bottleneck_channels,
            entropy_bottleneck=EntropyBottleneck(
                num_bottleneck_channels,
                tail_mass=1e-4,
            ),
        )


@register_model("um-pcc-simple-tmp")
class SimpleTmpPccModel(BasePccModel):
    def __init__(self, num_points=2048, num_bottleneck_channels=2048 * 3, gain=10):
        super().__init__()

        assert num_points * 3 % num_bottleneck_channels == 0

        # Perhaps "group" in some way, and then conv on that sub group?
        # Regroup to different levels...?

        self.g_a = nn.Sequential(
            conv1d(3 * num_points // num_bottleneck_channels, 64, 1),
            Reshape(
                (num_bottleneck_channels, 3 * num_points // num_bottleneck_channels)
            ),
            Gain((1,), gain),
        )

        self.g_s = nn.Sequential(
            Gain((1,), 1 / gain),
            Reshape((3, num_points)),
            Transpose(-2, -1),
        )

        self.latent_codec = EntropyBottleneckLatentCodec(
            N=num_bottleneck_channels,
            entropy_bottleneck=EntropyBottleneck(
                num_bottleneck_channels,
                tail_mass=1e-4,
            ),
        )


# TODO maybe the problem is when we don't give the bottleneck enough samples?
# TODO retry other models (including PointNet-PCC paper), but with a SINGLE channel bottleneck!
# TODO also add a 10 * gain vector
# TODO also, infer every n epochs... faster when prototyping...
# TODO num_bottleneck_channels, num_elem_per_bottleneck_channel
# TODO bottleneck_shape = (num_bottleneck_channels, num_elem_per_bottleneck_channel)


@register_model("um-pcc-simple-mlp")
class SimpleMlpPccModel(BasePccModel):
    def __init__(self, num_points=2048, num_bottleneck_channels=2048):
        super().__init__()

        assert num_points == num_bottleneck_channels

        groups = 128

        self.g_a = nn.Sequential(
            Reshape((3 * num_points, 1)),
            conv1d(3 * num_points, num_bottleneck_channels, 1, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_bottleneck_channels),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_bottleneck_channels),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            Gain((num_bottleneck_channels, 1), GAIN),
        )

        self.g_s = nn.Sequential(
            Gain((num_bottleneck_channels, 1), 1 / GAIN),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_bottleneck_channels),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_bottleneck_channels),
            conv1d(num_bottleneck_channels, 3 * num_points, 1, groups=groups),
            Reshape((num_points, 3)),
        )

        self.latent_codec = EntropyBottleneckLatentCodec(
            N=num_bottleneck_channels,
            entropy_bottleneck=EntropyBottleneck(
                num_bottleneck_channels,
                tail_mass=1e-4,
            ),
        )
