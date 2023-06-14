from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.ulhaqm import Interleave, NamedLayer, Reshape, conv1d
from src.layers.ulhaqm.pcc import (
    ClusterAttention,
    pointnet_classification_backend,
    pointnet_g_a_simple,
    pointnet_g_s_simple,
)


class BaseMultitaskPccModel(CompressionModel):
    latent_codec: Mapping[str, LatentCodec]

    def _setup_hooks(self):
        def hook(module, input, output):
            self.outputs[module.name] = output

        for _, module in self.task_backend.named_modules():
            if not isinstance(module, NamedLayer):
                continue
            module.register_forward_hook(hook)

    def forward(self, input):
        self.outputs = {}
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y1, y2 = self.split(y)
        y1_out = self.latent_codec["y1"](y1)
        y2_out = self.latent_codec["y2"](y2)
        y1_hat = y1_out["y_hat"]
        y2_hat = y2_out["y_hat"]
        t_hat = self.task_backend(y1_hat)
        y_hat = self.merge(y1_hat, y2_hat)
        y_hat_rec = y_hat.detach() if self.detach_y_hat else y_hat
        x_hat = self.g_s(y_hat_rec)
        assert x_hat.shape == x.shape

        return {
            "x_hat": x_hat,
            "t_hat": t_hat,
            "likelihoods": {
                "y1": y1_out["likelihoods"]["y"],
                "y2": y2_out["likelihoods"]["y"],
            },
            **{k: v for k, v in self.outputs.items()},
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        self.outputs = {}
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y1, y2 = self.split(y)
        y1_out = self.latent_codec["y1"].compress(y1)
        y2_out = self.latent_codec["y2"].compress(y2)
        [y1_strings] = y1_out["strings"]
        [y2_strings] = y2_out["strings"]
        # NOTE: the shape in y*_out["shape"] is incorrectly y.shape[-2:]
        return {"strings": [y1_strings, y2_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        self.outputs = {}
        [y1_strings, y2_strings] = strings
        y1_out = self.latent_codec["y1"].decompress([y1_strings], shape)
        y2_out = self.latent_codec["y2"].decompress([y2_strings], shape)
        y1_hat = y1_out["y_hat"]
        y2_hat = y2_out["y_hat"]
        t_hat = self.task_backend(y1_hat)
        y_hat = self.merge(y1_hat, y2_hat)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "t_hat": t_hat}

    def split(self, y):
        y1 = y[:, : self.num_split_channels[0]]
        y2 = y[:, self.num_split_channels[0] :]
        return y1, y2

    def merge(self, *args):
        return torch.cat(args, dim=1)


@register_model("um-pcc-multitask-example")
class ExampleMultitaskPccModel(BaseMultitaskPccModel):
    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=1024,
        num_split_channels=[1024, 1],
        num_classes=40,
        num_channels={
            "g_a": {
                "pointwise": [3, 64, 64, 64, 128, 1024],
                "mixer": [1024, 1025],
            },
            "g_s": [1025, 256, 512, 1024 * 3],
            "task_backend": [1024, 512, 256, 40],
        },
        detach_y_hat=True,
    ):
        super().__init__()

        self.detach_y_hat = detach_y_hat

        num_channels_g_a = [
            *num_channels["g_a"]["pointwise"],
            *num_channels["g_a"]["mixer"][1:],
        ]

        assert num_channels["task_backend"][0] == num_split_channels[0]
        assert num_channels["task_backend"][-1] == num_classes
        assert num_channels["g_s"][-1] == num_points * 3
        assert num_channels["g_s"][0] == num_channels_g_a[-1] == sum(num_split_channels)

        self.num_split_channels = num_split_channels

        self.g_a = pointnet_g_a_simple(num_channels["g_a"])

        self.g_s = pointnet_g_s_simple(num_channels["g_s"])

        # TODO self.task_frontend (for target[f"s_{}"])

        self.task_backend = pointnet_classification_backend(
            num_channels=num_channels["task_backend"],
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y1": EntropyBottleneckLatentCodec(
                    N=num_split_channels[0],
                    entropy_bottleneck=EntropyBottleneck(
                        num_split_channels[0], tail_mass=1e-4
                    ),
                ),
                "y2": EntropyBottleneckLatentCodec(
                    N=num_split_channels[1],
                    entropy_bottleneck=EntropyBottleneck(
                        num_split_channels[1], tail_mass=1e-4
                    ),
                ),
            }
        )

        self.outputs = {}
        self._setup_hooks()


@register_model("um-pcc-multitask-mlp-resmlp-clusterattn")
class ResMlpClusterAttnMlpPccModel(BaseMultitaskPccModel):
    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self,
        num_points=2048,
        num_bottleneck_channels=512,
        num_split_channels=[256, 256],
        num_classes=40,
        detach_y_hat=True,
    ):
        super().__init__()

        self.detach_y_hat = detach_y_hat

        # g_a_num_channels = [
        #     3,
        #     64,
        #     128,
        #     128,
        #     256,
        #     num_bottleneck_channels,
        #     num_bottleneck_channels,
        # ]

        g_s_num_channels = [
            num_bottleneck_channels,
            256,
            256,
            num_points * 3,
        ]

        assert sum(num_split_channels) == num_bottleneck_channels
        self.num_split_channels = num_split_channels

        # make_act = lambda: nn.ReLU(inplace=True)
        # make_act = lambda: nn.GELU()
        make_act = lambda: nn.Mish(inplace=True)

        # TODO This is quite deep. Use residual connections?

        groups = 16

        self.g_a = nn.Sequential(
            conv1d(3, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            make_act(),
            #
            ClusterAttention(32, num_points // 2, 8),
            # nn.BatchNorm1d(32),
            # make_act(),
            #
            conv1d(32, 64, 15),
            nn.BatchNorm1d(64),
            make_act(),
            #
            nn.MaxPool1d(3, stride=2, padding=1),
            #
            ClusterAttention(64, num_points // 4, 8),
            # nn.BatchNorm1d(64),
            # make_act(),
            #
            conv1d(64, 64, 15, stride=2),
            nn.BatchNorm1d(64),
            make_act(),
            #
            ClusterAttention(64, num_points // 8, 16),
            # nn.BatchNorm1d(64),
            # make_act(),
            #
            conv1d(64, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            make_act(),
            #
            nn.MaxPool1d(3, stride=2, padding=1),
            #
            Reshape((32 * num_points // 32, 1)),
            #
            conv1d(num_points, num_bottleneck_channels, 1, groups=groups),
            nn.BatchNorm1d(num_bottleneck_channels),
            make_act(),
            #
            Interleave(groups),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.BatchNorm1d(num_bottleneck_channels),
            make_act(),
            #
            Interleave(groups),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.BatchNorm1d(num_bottleneck_channels),
            make_act(),
            #
            Interleave(groups),
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
            nn.BatchNorm1d(num_bottleneck_channels),
            make_act(),
            #
            conv1d(num_bottleneck_channels, num_bottleneck_channels, 1, groups=groups),
        )

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

        self.task_backend = pointnet_classification_backend(
            num_channels=[num_split_channels[0], 512, 256, num_classes],
        )

        self.latent_codec = nn.ModuleDict(
            {
                "y1": EntropyBottleneckLatentCodec(
                    N=num_split_channels[0],
                    entropy_bottleneck=EntropyBottleneck(
                        num_split_channels[0], tail_mass=1e-4
                    ),
                ),
                "y2": EntropyBottleneckLatentCodec(
                    N=num_split_channels[1],
                    entropy_bottleneck=EntropyBottleneck(
                        num_split_channels[1], tail_mass=1e-4
                    ),
                ),
            }
        )

        self.outputs = {}
        self._setup_hooks()
