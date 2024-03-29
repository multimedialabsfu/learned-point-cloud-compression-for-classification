import os
import shutil
from pathlib import Path

from torch.utils.data import ConcatDataset

from compressai.registry import register_dataset
from src.datasets.ndarray import NdArrayDataset
from src.datasets.stack import StackDataset

from .cache import CacheDataset
from .utils import download_url, hash_file


@register_dataset("S3disDataset")
class S3disDataset(CacheDataset):
    """S3DIS dataset.

    The Stanford 3D Indoor Scene Dataset (S3DIS) dataset, introduced by
    [Armeni2012]_, contains 3D point clouds of 6 large-scale indoor areas.
    There are multiple rooms (e.g. office, lounge, hallway, etc) per area.
    See the [ProjectPage]_ for a visualization.

    The ``semantic_index`` is a number between 0 and 12 (inclusive),
    which can be used as the semantic label for each point.

    References:

        .. [Armeni2012] `"3D Semantic Parsing of Large-Scale Indoor Spaces,"
           <https://openaccess.thecvf.com/content_cvpr_2016/html/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.html>`_,
           by Iro Armeni, Ozan Sener, Amir R. Zamir, Helen Jiang,
           Ioannis Brilakis, Martin Fischer, and Silvio Savarese,
           CVPR 2012.

        .. [ProjectPage] `Project page
        <http://buildingparser.stanford.edu/dataset.html>`_

        .. [PapersWithCode] `PapersWithCode: S3DIS
        <https://paperswithcode.com/dataset/s3dis>`_
    """

    URLS = [
        "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip",
    ]

    HASHES = [
        "587bb63b296d542c24910c384c41028f2caa1d749042ae891d0d64968c773185",  # indoor3d_sem_seg_hdf5_data.zip
    ]

    # Suggested splits:
    AREAS = {
        "train": (1, 2, 3, 4, 6),
        "valid": (5,),
        "test": (5,),
    }

    NUM_SAMPLES_PER_AREA = [0, 3687, 4440, 1650, 3662, 6852, 3294]

    LABELS = [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ]

    ROOMS = [
        "auditorium",
        "conferenceRoom",
        "copyRoom",
        "hallway",
        "lobby",
        "lounge",
        "office",
        "openspace",
        "pantry",
        "storage",
        "WC",
    ]

    def __init__(
        self,
        root=None,
        cache_root=None,
        split="train",
        split_name=None,
        areas=AREAS["train"],
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
        self.areas = areas

        if download and self.root:
            self.download()

        self._root_dataset = self._get_root_dataset()

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
        for expected_hash, url in zip(self.HASHES, self.URLS):
            filepath = download_url(
                url, tmpdir, check_certificate=False, overwrite=force
            )
            shutil.unpack_archive(filepath, tmpdir)
            assert expected_hash == hash_file(filepath, method="sha256")
        shutil.move(tmpdir / "indoor3d_sem_seg_hdf5_data", self.root)

    def _get_root_dataset(self):
        import h5py

        h5_files = [h5py.File(path, "r") for path in sorted(self.root.glob("**/*.h5"))]
        keys = ["data", "label"]

        return ConcatDataset(
            StackDataset(**{k: NdArrayDataset(h5_file[k], single=True) for k in keys})
            for h5_file in h5_files
        )

    def _get_items(self):
        with open(self.root / "room_filelist.txt") as f:
            lines = f.read().splitlines()
        return [
            (i, line)
            for i, line in enumerate(lines)
            if int(line.split("_")[1]) in self.areas
        ]

    def _load_item(self, item):
        index, name = item
        _, area_index_str, room_str, *_ = name.split("_")
        data = self._root_dataset[index]

        return {
            "file_index": index,
            "area_index": int(area_index_str),
            "room_index": self.ROOMS.index(room_str),
            "semantic_index": data["label"],
            "pos": data["data"][:, 0:3],  # xyz
            "color": data["data"][:, 3:6],  # rgb
            "pos_normalized": data["data"][:, 6:9],  # Normalized xyz
        }
