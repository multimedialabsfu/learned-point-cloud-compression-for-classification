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
        rot = random_3x3_rotation_matrix(1).to(data.pos.device).squeeze(0)
        data.pos = data.pos @ rot.T
        return data


# See https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4832876#4832876
def random_3x3_rotation_matrix(batch_size: int, generator=None) -> torch.Tensor:
    z = torch.randn((batch_size, 3, 3), generator=generator)
    z = z / torch.linalg.norm(z, axis=-1, keepdims=True)
    z[:, 0] = torch.linalg.cross(z[:, 1], z[:, 2], axis=-1)
    z[:, 0] = z[:, 0] / torch.linalg.norm(z[:, 0], axis=-1, keepdims=True)
    z[:, 1] = torch.linalg.cross(z[:, 2], z[:, 0], axis=-1)
    z[:, 1] = z[:, 1] / torch.linalg.norm(z[:, 1], axis=-1, keepdims=True)
    return z
