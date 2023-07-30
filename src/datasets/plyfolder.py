from __future__ import annotations

import re
from pathlib import Path

from torch.utils.data import Dataset

from compressai.registry import register_dataset
from src.utils.point_cloud import pc_read

from .modelnet import PointCloudPreprocessor


@register_dataset("PlyFolderDataset")
class PlyFolderDataset(Dataset):
    """Dataset for point-clouds.

    File paths may take the following forms:

    .. code-block:: text

        ${LOADER}_${LABEL_INDEX}_${INDEX}*.ply
        train_00_000.ply
        train_00_000.bin.ply
        train_00_000.rec.ply
    """

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.paths = sorted(Path(root).glob("**/*.ply"))
        self.transform = transform
        self.preprocessor = PointCloudPreprocessor()

    def __getitem__(self, index):
        path = self.paths[index]
        pattern = (
            r"^(?P<loader>.*)_(?P<label_idx>\d+)_(?P<index>\d+)(\.bin|\.rec)?\.ply$"
        )
        match = re.match(pattern, path.name)
        if match is None:
            raise ValueError(f"Could not parse path: {path}")
        label_idx = int(match.group("label_idx"))
        points = pc_read(path)
        # assert points.shape[0] == self.num_points
        assert points.shape[1] == 3
        points = self.preprocessor.preprocess(points)
        points = self.transform(points)
        return {
            "index": index,
            "points": points,
            "labels": label_idx,
            # "path": str(path),
        }

    def __len__(self):
        return len(self.paths)
