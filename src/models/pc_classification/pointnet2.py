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
from src.layers.pc_pointnet2 import PointNetSetAbstraction


@register_model("um-pc-cls-pointnet2")
class PointNet2ClassPcModel(nn.Module):
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

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.down = nn.ModuleDict(
            {
                "_1": PointNetSetAbstraction(
                    npoint=512,
                    radius=0.2,
                    nsample=32,
                    in_channel=in_channel,
                    mlp=[64, 64, 128],
                    group_all=False,
                ),
                "_2": PointNetSetAbstraction(
                    npoint=128,
                    radius=0.4,
                    nsample=64,
                    in_channel=128 + 3,
                    mlp=[128, 128, 256],
                    group_all=False,
                ),
                "_3": PointNetSetAbstraction(
                    npoint=None,
                    radius=None,
                    nsample=None,
                    in_channel=256 + 3,
                    mlp=[256, 512, 1024],
                    group_all=True,
                ),
            }
        )

        self.task_backend = nn.Sequential(
            Reshape((1024,)),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            # nn.LogSoftmax(dim=-1),
            # NOTE: The log-softmax is done in nn.CrossEntropyLoss since
            # nn.CrossEntropyLoss == nn.NLLLoss ∘ nn.LogSoftmax,
            # where nn.NLLLoss doesn't acutally contain a log, despite its name.
        )

    def forward(self, input):
        points = input["points"].transpose(-2, -1)

        if self.normal_channel:
            xyz = points[:, :3, :]
            norm = points[:, 3:, :]
        else:
            xyz = points
            norm = None

        xyz_ = {0: xyz}
        u_ = {0: norm}

        for i in range(1, 4):
            xyz_[i], u_[i] = self.down[f"_{i}"](xyz_[i - 1], u_[i - 1])

        t_hat = self.task_backend(u_[3])

        return {
            "t_hat": t_hat,
            # Additional outputs:
            "debug_outputs": {},
        }
