import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("random_rotate_full")
@register_transform("RandomRotateFull")
class RandomRotateFull(BaseTransform):
    r"""Randomly rotates node positions around the origin
    (functional name: :obj:`random_rotate_full`).
    """

    def __call__(self, data: Data) -> Data:
        _, ndim = data.pos.shape
        rot = random_rotation_matrix(1, ndim).to(data.pos.device).squeeze(0)
        data.pos = data.pos @ rot.T
        return data


# See https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4832876#4832876
def random_rotation_matrix(batch_size: int, ndim=3, generator=None) -> torch.Tensor:
    z = torch.randn((batch_size, ndim, ndim), generator=generator)
    q, r = torch.linalg.qr(z)
    sign = 2 * (r.diagonal(dim1=-2, dim2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= torch.linalg.det(rot)[..., None]
    return rot
