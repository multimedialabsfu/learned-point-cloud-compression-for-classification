import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("sample_points_v2")
@register_transform("SamplePointsV2")
class SamplePointsV2(BaseTransform):
    r"""Uniformly samples a fixed number of points on the mesh faces according
    to their face area (functional name: :obj:`sample_points`).

    Adapted from PyTorch Geometric under MIT license at
    https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
        seed (int, optional): Initial random seed.
        static_seed (int, optional): Reset random seed to this every call.
    """

    def __init__(
        self,
        num: int,
        *,
        remove_faces: bool = True,
        include_normals: bool = False,
        seed=None,
        static_seed=None,
    ):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.static_seed = static_seed

    def __call__(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None

        if self.static_seed is not None:
            self.generator.manual_seed(self.static_seed)

        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device, generator=self.generator)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = torch.nn.functional.normalize(vec1.cross(vec2), p=2)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num})"
