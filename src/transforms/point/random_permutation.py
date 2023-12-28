import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("random_permutation")
@register_transform("RandomPermutation")
class RandomPermutation(BaseTransform):
    r"""Randomly permutes points and associated attributes
    (functional name: :obj:`random_permutation`).
    """

    def __init__(self, *, attrs=("pos",)):
        self.attrs = attrs

    def __call__(self, data: Data) -> Data:
        perm = torch.randperm(data.pos.shape[0])
        return Data(**{k: v[perm] if k in self.attrs else v for k, v in data.items()})
