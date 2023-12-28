from contextlib import suppress

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.registry import register_transform


@functional_transform("generate_position_normals")
@register_transform("GeneratePositionNormals")
class GeneratePositionNormals(BaseTransform):
    r"""Generates normals from node positions
    (functional name: :obj:`generate_position_normals`).
    """

    def __init__(self, *, method="any", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        assert data.pos.ndim == 2 and data.pos.shape[1] == 3

        if method == "open3d":
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data.pos.cpu().numpy())
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN())
            pcd.normalize_normals()
            data.norm = torch.tensor(
                pcd.normals, dtype=torch.float32, device=data.pos.device
            )

            return data

        if method == "pytorch3d":
            import pytorch3d.ops

            data.norm = pytorch3d.ops.estimate_pointcloud_normals(
                data.pos.unsqueeze(0), **self.kwargs
            ).squeeze(0)

            return data

        if method == "any":
            for method in ["open3d", "pytorch3d"]:
                with suppress(ImportError):
                    return self(data)
            raise RuntimeError("Please install open3d / pytorch3d to estimate normals.")

        raise ValueError(f"Unknown method: {self.method}")
