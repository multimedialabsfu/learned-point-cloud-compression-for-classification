from __future__ import annotations

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.models import CompressionModel
from compressai.registry import register_model
from src.layers.layers import Gain, Reshape, Transpose
from src.layers.pc_pointnet2 import PointNetSetAbstraction
from src.layers.pcc import GAIN
from src.layers.pcc_reconstruction.pointnet2 import UpsampleBlock


@register_model("um-pcc-rec-pointnet2")
class PointNet2ReconstructionPccModel(CompressionModel):
    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            # NOTE: Ignored for now.
        },
        normal_channel=False,
    ):
        super().__init__()

        self.normal_channel = bool(normal_channel)

        # Original PointNet++ architecture:
        # D = [3 * self.normal_channel, 128, 256, 1024]
        # P = [None, 512, 128, 1]
        # S = [None, 32, 64, 128]

        D = [3 * self.normal_channel, 128, 256, 1024]
        P = [num_points, 256, 64, 1]
        S = [None, 4, 4, 64]

        assert P[0] == P[1] * S[1]
        assert P[1] == P[2] * S[2]
        assert P[2] == P[3] * S[3]

        E = [3, 32, 32, 32, 0]
        M = [64, 128, 256, 1024]

        self.down = nn.ModuleDict(
            {
                "_1": PointNetSetAbstraction(
                    npoint=P[1],
                    radius=0.2,
                    nsample=S[1],
                    in_channel=D[0] + 3,
                    mlp=[64, 64, D[1]],
                    group_all=False,
                ),
                "_2": PointNetSetAbstraction(
                    npoint=P[2],
                    radius=0.4,
                    nsample=S[2],
                    in_channel=D[1] + 3,
                    mlp=[128, 128, D[2]],
                    group_all=False,
                ),
                "_3": PointNetSetAbstraction(
                    npoint=None,
                    radius=None,
                    nsample=None,
                    in_channel=D[2] + 3,
                    mlp=[256, 512, D[3]],
                    group_all=True,
                ),
            }
        )

        self.h_a = nn.ModuleDict(
            {
                "_0": nn.Sequential(
                    Reshape((D[0] + 3, P[1] * S[1])),
                    nn.Conv1d(D[0] + 3, M[0], 1),
                    Gain((M[0], 1), factor=GAIN),
                ),
                "_1": nn.Sequential(
                    Reshape((D[1] + 3, P[2] * S[2])),
                    nn.Conv1d(D[1] + 3, M[1], 1),
                    Gain((M[1], 1), factor=GAIN),
                ),
                "_2": nn.Sequential(
                    Reshape((D[2] + 3, P[3] * S[3])),
                    nn.Conv1d(D[2] + 3, M[2], 1),
                    Gain((M[2], 1), factor=GAIN),
                ),
                "_3": nn.Sequential(
                    Reshape((D[3], 1)),
                    nn.Conv1d(D[3], M[3], 1),
                    Gain((M[3], 1), factor=GAIN),
                ),
            }
        )

        self.h_s = nn.ModuleDict(
            {
                "_0": nn.Sequential(
                    Gain((M[0], 1), factor=1 / GAIN),
                    nn.Conv1d(M[0], D[0] + 3, 1),
                ),
                "_1": nn.Sequential(
                    Gain((M[1], 1), factor=1 / GAIN),
                    nn.Conv1d(M[1], D[1] + 3, 1),
                ),
                "_2": nn.Sequential(
                    Gain((M[2], 1), factor=1 / GAIN),
                    nn.Conv1d(M[2], D[2] + 3, 1),
                ),
                "_3": nn.Sequential(
                    Gain((M[3], 1), factor=1 / GAIN),
                    nn.Conv1d(M[3], D[3], 1),
                ),
            }
        )

        self.up = nn.ModuleDict(
            {
                "_0": nn.Sequential(
                    nn.Conv1d(E[1] + D[0] + 3, E[1], 1),
                    nn.BatchNorm1d(E[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(E[1], E[0], 1),
                    Reshape((E[0], P[0])),
                    Transpose(-2, -1),
                ),
                "_1": UpsampleBlock(D, E, P, S, i=1, extra_in_ch=3, groups=(1, 4)),
                "_2": UpsampleBlock(D, E, P, S, i=2, extra_in_ch=3, groups=(1, 4)),
                "_3": UpsampleBlock(D, E, P, S, i=3, extra_in_ch=0, groups=(1, 32)),
            }
        )

        self.latent_codec = nn.ModuleDict(
            {
                "_0": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(M[0], tail_mass=1e-4),
                ),
                "_1": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(M[1], tail_mass=1e-4),
                ),
                "_2": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(M[2], tail_mass=1e-4),
                ),
                "_3": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(M[3], tail_mass=1e-4),
                ),
            }
        )

    def forward(self, input):
        points = input["points"].transpose(-2, -1)

        if self.normal_channel:
            xyz = points[:, :3, :]
            norm = points[:, 3:, :]
        else:
            xyz = points
            norm = None

        # TODO Describe these.
        xyz_ = {0: xyz}
        u_ = {0: norm}
        uu_ = {}
        uu_hat_ = {}
        y_ = {}
        y_hat_ = {}
        y_out_ = {}
        v_ = {}

        for i in range(1, 4):
            down_out_i = self.down[f"_{i}"](xyz_[i - 1], u_[i - 1])
            xyz_[i] = down_out_i["new_xyz"]
            u_[i] = down_out_i["new_features"]
            uu_[i - 1] = down_out_i["grouped_features"]

        uu_[3] = u_[3][:, :, None, :]

        for i in reversed(range(0, 4)):
            y_[i] = self.h_a[f"_{i}"](uu_[i])
            y_out_[i] = self.latent_codec[f"_{i}"](y_[i])
            y_hat_[i] = y_out_[i]["y_hat"]
            uu_hat_[i] = self.h_s[f"_{i}"](y_hat_[i])

        B, _, *tail = uu_hat_[3].shape
        v_[4] = uu_hat_[3].new_zeros((B, 0, *tail))

        for i in reversed(range(0, 4)):
            v_[i] = self.up[f"_{i}"](torch.cat([v_[i + 1], uu_hat_[i]], dim=1))

        x_hat = v_[0]

        return {
            "x_hat": x_hat,
            "likelihoods": {f"y_{i}": y_out_[i]["likelihoods"]["y"] for i in range(4)},
            # Additional outputs:
            # "y": y,
            # "y_hat": y_hat,
            "debug_outputs": {
                # "y_hat": y_hat,
            },
        }

    def compress(self, input):
        raise NotImplementedError

    def decompress(self, strings, shape):
        raise NotImplementedError
