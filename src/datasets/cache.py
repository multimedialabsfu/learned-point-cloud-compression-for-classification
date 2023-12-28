import json
import os
import os.path
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class CacheDataset(Dataset):
    def __init__(
        self,
        cache_root=None,
        pre_transform=None,
        transform=None,
    ):
        self.__cache_root = Path(cache_root)
        self.pre_transform = pre_transform
        self.transform = transform
        self._store = {}

    def __len__(self):
        return len(self._store[next(iter(self._store))])

    def __getitem__(self, index):
        data = {k: v[index].copy() for k, v in self._store.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _ensure_cache(self):
        try:
            self._load_cache(mode="r")
        except FileNotFoundError:
            self._generate_cache()
            self._load_cache(mode="r")

    def _load_cache(self, mode):
        with open(self.__cache_root / "info.json", "r") as f:
            info = json.load(f)

        self._store = {
            k: np.memmap(
                self.__cache_root / f"{k}.npy",
                mode=mode,
                dtype=settings["dtype"],
                shape=tuple(settings["shape"]),
            )
            for k, settings in info.items()
        }

    def _generate_cache(self, verbose=True):
        if verbose:
            print(f"Generating cache at {self.__cache_root}...")

        items = self._get_items()

        if verbose:
            items = tqdm(items)

        for i, item in enumerate(items):
            data = self._load_item(item)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if not self._store:
                self._write_cache_info(len(items), data)
                self._load_cache(mode="w+")

            for k, v in data.items():
                self._store[k][i] = v

    def _write_cache_info(self, num_samples, data):
        info = {
            k: {
                "dtype": _removeprefix(str(v.dtype), "torch."),
                "shape": (num_samples, *v.shape),
            }
            for k, v in data.items()
        }
        os.makedirs(self.__cache_root, exist_ok=True)
        with open(self.__cache_root / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _get_items(self):
        raise NotImplementedError

    def _load_item(self, item):
        raise NotImplementedError


def _removeprefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s
