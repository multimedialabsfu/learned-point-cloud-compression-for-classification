import torch.nn as nn
from torch import Tensor

from src.layers.layers import Interleave, Reshape, Transpose


class UpsampleBlock(nn.Module):
    def __init__(self, D, E, P, S, i, extra_in_ch=3, groups=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(E[i + 1] + D[i] + extra_in_ch, D[i], 1, groups=groups[0]),
            Interleave(groups=groups[0]),
            nn.BatchNorm1d(D[i]),
            nn.ReLU(inplace=True),
            nn.Conv1d(D[i], E[i] * S[i], 1, groups=groups[1]),
            Interleave(groups=groups[1]),
            nn.BatchNorm1d(E[i] * S[i]),
            nn.ReLU(inplace=True),
            Reshape((E[i], S[i], P[i])),
            Transpose(-2, -1),
            Reshape((E[i], P[i] * S[i])),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
