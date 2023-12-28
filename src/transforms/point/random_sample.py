import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("random_sample")
@register_transform("RandomSample")
class RandomSample(BaseTransform):
    r"""Randomly samples points and associated attributes
    (functional name: :obj:`random_sample`).
    """

    def __init__(
        self,
        num=None,
        *,
        attrs=("pos",),
        remove_duplicates_by=None,
        preserve_order=False,
        seed=None,
        static_seed=None,
    ):
        self.num = num
        self.attrs = attrs
        self.remove_duplicates_by = remove_duplicates_by
        self.preserve_order = preserve_order
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.static_seed = static_seed

    def __call__(self, data: Data) -> Data:
        if self.static_seed is not None:
            self.generator.manual_seed(self.static_seed)

        if self.remove_duplicates_by is not None:
            _, perm = data[self.remove_duplicates_by].unique(return_inverse=True, dim=0)
            for attr in self.attrs:
                data[attr] = data[attr][perm]

        num_input = data[self.attrs[0]].shape[0]
        assert all(data[k].shape[0] == num_input for k in self.attrs)

        p = torch.ones(max(num_input, self.num), dtype=torch.float32)
        perm = torch.multinomial(p, self.num, generator=self.generator)
        perm %= num_input

        if self.preserve_order:
            perm = perm.sort()[0]

        return Data(**{k: v[perm] if k in self.attrs else v for k, v in data.items()})
