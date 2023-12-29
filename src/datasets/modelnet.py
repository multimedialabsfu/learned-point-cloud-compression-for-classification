import os
import os.path
import re
import shutil
from pathlib import Path

import numpy as np
from pyntcloud import PyntCloud

from compressai.registry import register_dataset

from .cache import CacheDataset
from .utils import download_url, hash_file


@register_dataset("ModelNetDataset")
class ModelNetDataset(CacheDataset):
    """ModelNet dataset.

    This dataset of 3D CAD models of objects was introduced by
    [Wu2015]_, consisting of 10 or 40 classes, with 4899 and 12311
    aligned items, respectively.
    Each 3D model is represented in the OFF file format by a triangle
    mesh (i.e. faces) and has a single label (e.g. airplane).
    To convert the triangle meshes to point clouds, one may use a mesh
    sampling method (e.g. ``SamplePoints``).

    References:

        .. [Wu2015] `"3D ShapeNets: A deep representation for volumetric
           shapes," <https://arxiv.org/abs/1406.5670>`_, by Zhirong Wu,
           Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou
           Tang, and Jianxiong Xiao, CVPR 2015.
    """

    # fmt: off
    LABEL_LIST = {
        "10": [
            "bathtub", "bed", "chair", "desk", "dresser",
            "monitor", "night_stand", "sofa", "table", "toilet",
        ],
        "40": [
            "airplane", "bathtub", "bed", "bench", "bookshelf",
            "bottle", "bowl", "car", "chair", "cone", "cup",
            "curtain", "desk", "door", "dresser", "flower_pot",
            "glass_box", "guitar", "keyboard", "lamp", "laptop",
            "mantel", "monitor", "night_stand", "person", "piano",
            "plant", "radio", "range_hood", "sink", "sofa",
            "stairs", "stool", "table", "tent", "toilet",
            "tv_stand", "vase", "wardrobe", "xbox",
        ],
    }
    # fmt: on

    LABEL_STR_TO_LABEL_INDEX = {
        "10": {label: idx for idx, label in enumerate(LABEL_LIST["10"])},
        "40": {label: idx for idx, label in enumerate(LABEL_LIST["40"])},
    }

    URLS = {
        "10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    }

    HASHES = {
        "10": "9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51",
        "40": "42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac",
    }

    def __init__(
        self,
        root=None,
        cache_root=None,
        split="train",
        split_name=None,
        name="40",
        pre_transform=None,
        transform=None,
        download=True,
    ):
        if cache_root is None:
            assert root is not None
            cache_root = f"{str(root).rstrip('/')}_cache"

        self.root = Path(root) if root else None
        self.cache_root = Path(cache_root)
        self.split = split
        self.split_name = split if split_name is None else split_name
        self.name = name

        if download and self.root:
            self.download()

        super().__init__(
            cache_root=self.cache_root / self.split_name,
            pre_transform=pre_transform,
            transform=transform,
        )

        self._ensure_cache()

    def download(self, force=False):
        if not force and self.root.exists():
            return
        tmpdir = self.root.parent / "tmp"
        os.makedirs(tmpdir, exist_ok=True)
        filepath = download_url(self.URLS[self.name], tmpdir, overwrite=force)
        assert self.HASHES[self.name] == hash_file(filepath, method="sha256")
        shutil.unpack_archive(filepath, tmpdir)
        shutil.move(tmpdir / f"ModelNet{self.name}", self.root)

    def _get_items(self):
        return sorted(self.root.glob(f"**/{self.split}/*.off"))

    def _load_item(self, path):
        label_index, file_index = self._parse_path(path)
        cloud = PyntCloud.from_file(str(path))
        return {
            "file_index": np.array([file_index], dtype=np.int32),
            "label": np.array([label_index], dtype=np.uint8),
            "pos": cloud.points.values,
            "face": cloud.mesh.values.T,
        }

    def _parse_path(self, path):
        pattern = (
            r"^.*?/?"
            r"(?P<label_str>[a-zA-Z_]+)/"
            r"(?P<split>[a-zA-Z_]+)/"
            r"(?P<label_str_again>[a-zA-Z_]+)_(?P<file_index>\d+)\.off$"
        )
        match = re.match(pattern, str(path))
        if match is None:
            raise ValueError(f"Could not parse path: {path}")
        assert match.group("split") == self.split
        assert match.group("label_str") == match.group("label_str_again")
        label_str = match.group("label_str")
        label_index = self.LABEL_STR_TO_LABEL_INDEX[self.name][label_str]
        file_index = int(match.group("file_index"))
        return label_index, file_index
