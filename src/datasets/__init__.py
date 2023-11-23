import compressai.datasets as _M
from compressai.datasets import *

from .modelnet import ModelNetSimple
from .plyfolder import PlyFolderDataset

__all__ = [
    *_M.__all__,
    "ModelNetSimple",
    "PlyFolderDataset",
]
