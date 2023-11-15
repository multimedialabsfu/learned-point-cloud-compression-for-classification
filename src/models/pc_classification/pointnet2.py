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
import torch.nn.functional as F

from compressai.registry import register_model
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
        self.sa1 = PointNetSetAbstraction(
            npoint=num_points,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, input):
        xyz = input["points"].transpose(-2, -1)
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)  # NOTE: This is done in nn.CrossEntropyLoss.
        # In fact, nn.CrossEntropyLoss == nn.NLLLoss ∘ nn.LogSoftmax,
        # where nn.NLLLoss doesn't acutally contain a log, despite its name.
        t_hat = x

        # return x, l3_points
        return {
            "t_hat": t_hat,
            # **{k: v for k, v in self.outputs.items()},
            # Additional outputs:
            # "y": y,
            "debug_outputs": {},
        }
