from typing import Any, Dict

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("to_dict")
@register_transform("ToDict")
class ToDict(BaseTransform):
    r"""Convert :obj:`Mapping[str, Any]`
    (functional name: :obj:`to_dict`).
    """

    def __init__(self, *, wrapper="dict"):
        if wrapper == "dict":
            self.wrap = dict
        elif wrapper == "torch_geometric.data.Data":
            self.wrap = Data
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

    def __call__(self, data) -> Dict[str, Any]:
        data = {
            k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
            for k, v in data.items()
        }
        return self.wrap(**data)
