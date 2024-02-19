# NOTE: This module has been adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
#
# See https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/LICENSE.
# LICENSE is also reproduced below:
#
#
# MIT License
#
# Copyright (c) 2019 benny
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch.nn as nn

from compressai.registry import register_model
from src.layers.layers import Reshape
from src.layers.pointnet2 import PointNetSetAbstraction


@register_model("sfu-pc-cls-pointnet2-ssg")
@register_model("um-pc-cls-pointnet2-ssg")
@register_model("um-pc-cls-pointnet2")
class PointNet2SsgClassPcModel(nn.Module):
    """PointNet++ classification model.

    Model by [Qi2017]_.
    Uses single-scale grouping (SSG) for point set abstraction.

    References:

        .. [Qi2017] `"PointNet++: Deep Hierarchical Feature Learning on
            Point Sets in a Metric Space"
            <https://arxiv.org/abs/1706.02413>`_, by Charles R. Qi,
            Li Yi, Hao Su, and Leonidas J. Guibas, NIPS 2017.
    """

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        num_channels={
            # NOTE: Ignored for now.
        },
        D=(0, 128, 256, 1024),
        P=(None, 512, 128, 1),
        S=(None, 32, 64, 128),
        R=(None, 0.2, 0.4, None),
        normal_channel=False,
    ):
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.D = D
        self.P = P
        self.S = S
        self.R = R
        self.normal_channel = bool(normal_channel)

        # Original PointNet++ architecture:
        # D = [3 * self.normal_channel, 128, 256, 1024]
        # P = [None, 512, 128, 1]
        # S = [None, 32, 64, 128]
        # R = [None, 0.2, 0.4, None]

        # NOTE: P[0] is only used to determine the number of output points.
        # assert P[0] == num_points

        # Disable asserts which are only needed by proposed rec model:
        # assert P[0] == P[1] * S[1]
        # assert P[1] == P[2] * S[2]
        # assert P[2] == P[3] * S[3]

        self.levels = 4

        self.down = nn.ModuleDict(
            {
                "_1": PointNetSetAbstraction(
                    npoint=P[1],
                    radius=R[1],
                    nsample=S[1],
                    in_channel=D[0] + 3,
                    mlp=[D[1] // 2, D[1] // 2, D[1]],
                    group_all=False,
                ),
                "_2": PointNetSetAbstraction(
                    npoint=P[2],
                    radius=R[2],
                    nsample=S[2],
                    in_channel=D[1] + 3,
                    mlp=[D[1], D[1], D[2]],
                    group_all=False,
                ),
                "_3": PointNetSetAbstraction(
                    npoint=None,
                    radius=None,
                    nsample=None,
                    in_channel=D[2] + 3,
                    mlp=[D[2], D[3], D[3]],
                    group_all=True,
                ),
            }
        )

        self.task_backend = nn.Sequential(
            nn.Conv1d(D[3], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(256, num_classes, 1),
            Reshape((num_classes,)),
            # nn.LogSoftmax(dim=-1),
            # NOTE: The log-softmax is done in nn.CrossEntropyLoss since
            # nn.CrossEntropyLoss == nn.NLLLoss ∘ nn.LogSoftmax,
            # where nn.NLLLoss doesn't acutally contain a log, despite its name.
        )

    def _get_inputs(self, input):
        points = input["pos"].transpose(-2, -1)
        if self.normal_channel:
            xyz = points[:, :3, :]
            norm = points[:, 3:, :]
        else:
            xyz = points
            norm = None
        return xyz, norm

    def forward(self, input):
        xyz, norm = self._get_inputs(input)

        xyz_ = {0: xyz}
        u_ = {0: norm}

        for i in range(1, 4):
            down_out_i = self.down[f"_{i}"](xyz_[i - 1], u_[i - 1])
            xyz_[i] = down_out_i["new_xyz"]
            u_[i] = down_out_i["new_features"]

        t_hat = self.task_backend(u_[3])

        return {
            "t_hat": t_hat,
            # Additional outputs:
            "debug_outputs": {},
        }
