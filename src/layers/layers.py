from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __repr__(self):
        return f"{self.__class__.__name__}(func={self.func})"

    def forward(self, x):
        return self.func(x)


class NamedLayer(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def forward(self, x):
        output_shape = (x.shape[0], *self.shape)
        try:
            return x.reshape(output_shape)
        except RuntimeError as e:
            e.args += (f"Cannot reshape input {tuple(x.shape)} to {output_shape}",)
            raise e


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def __repr__(self):
        return f"{self.__class__.__name__}(dim0={self.dim0}, dim1={self.dim1})"

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1).contiguous()


class Interleave(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        g = self.groups
        n, c, *tail = x.shape
        return x.reshape(n, g, c // g, *tail).transpose(1, 2).reshape(x.shape)


class Gain(nn.Module):
    def __init__(self, shape=None, factor: float = 1.0):
        super().__init__()
        self.factor = factor
        self.gain = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor) -> Tensor:
        return self.factor * self.gain * x


def conv1d(in_channels: int, out_channels: int, kernel_size: int, **kwargs):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        **kwargs,
    )
