import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, Center

from src.registry import register_transform


@functional_transform("normalize_scale_v2")
@register_transform("NormalizeScaleV2")
class NormalizeScaleV2(BaseTransform):
    r"""Centers and normalizes node positions
    (functional name: :obj:`normalize_scale_v2`).
    """

    def __init__(self, *, center=True, scale_method="linf"):
        self.scale_method = scale_method
        self.center = Center() if center else lambda x: x

    def __call__(self, data: Data) -> Data:
        data = self.center(data)
        data.pos = data.pos / self._compute_scale(data)
        return data

    def _compute_scale(self, data: Data) -> torch.Tensor:
        if self.scale_method == "l2":
            return (data.pos**2).sum(axis=-1).sqrt().max()
        if self.scale_method == "linf":
            return data.pos.abs().max()
        raise ValueError(f"Unknown scale_method: {self.scale_method}")
