from __future__ import annotations

import torch.nn as nn

from .layers import Gain, Interleave, NamedLayer, Reshape

GAIN = 10.0


def conv1d_group_seq(
    num_channels, groups, enabled=("bn", "act"), enabled_final=("bn", "act")
):
    assert len(num_channels) == 0 or len(groups) == len(num_channels) - 1
    xs = []
    for i in range(len(num_channels) - 1):
        is_final = i + 1 == len(num_channels) - 1
        xs.append(nn.Conv1d(num_channels[i], num_channels[i + 1], 1, groups=groups[i]))
        # ChannelShuffle is only required between consecutive group convs.
        if not is_final and groups[i] > 1 and groups[i + 1] > 1:
            xs.append(Interleave(groups[i]))
        if "bn" in enabled and (not is_final or "bn" in enabled_final):
            xs.append(nn.BatchNorm1d(num_channels[i + 1]))
        if "act" in enabled and (not is_final or "act" in enabled_final):
            xs.append(nn.ReLU(inplace=True))
    return nn.Sequential(*xs)


def pointnet_g_a_simple(num_channels, groups=None, gain=GAIN):
    if groups is None:
        groups = [1] * (len(num_channels) - 1)
    return nn.Sequential(
        *conv1d_group_seq(num_channels, groups),
        nn.AdaptiveMaxPool1d(1),
        Gain((num_channels[-1], 1), gain),
    )


def pointnet_g_s_simple(num_channels, gain=GAIN):
    num_points = num_channels[-1] // 3
    return nn.Sequential(
        Gain((num_channels[0], 1), 1 / gain),
        *[
            x
            for ch_in, ch_out in zip(num_channels[:-2], num_channels[1:-1])
            for x in [
                nn.Conv1d(ch_in, ch_out, 1),
                nn.ReLU(inplace=True),
            ]
        ],
        nn.Conv1d(num_channels[-2], num_channels[-1], 1),
        Reshape((num_points, 3)),
    )


def pointnet_classification_backend(num_channels):
    num_classes = num_channels[-1]
    return nn.Sequential(
        *[
            x
            for i, ch_in, ch_out in zip(
                range(len(num_channels) - 2),
                num_channels[:-2],
                num_channels[1:-1],
            )
            for x in [
                nn.Conv1d(ch_in, ch_out, 1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                NamedLayer(f"s_{i + 1}_hat"),
            ]
        ],
        nn.Dropout(0.3),
        nn.Conv1d(num_channels[-2], num_channels[-1], 1),
        Reshape((num_classes,)),
    )
