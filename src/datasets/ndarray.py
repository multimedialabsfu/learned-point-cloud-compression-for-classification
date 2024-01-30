# Adapted via https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/data/dataset.py
# BSD-style license: https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE

from typing import Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class NdArrayDataset(Dataset[Union[np.ndarray, Tuple[np.ndarray, ...]]]):
    r"""Dataset wrapping arrays.

    Each sample will be retrieved by indexing arrays along the first dimension.

    Args:
        *arrays (np.ndarray): arrays that have the same size of the first dimension.
    """

    arrays: Tuple[np.ndarray, ...]

    def __init__(self, *arrays: np.ndarray, single: bool = False) -> None:
        assert all(
            arrays[0].shape[0] == array.shape[0] for array in arrays
        ), "Size mismatch between arrays"
        self.arrays = arrays
        self.single = single

    def __getitem__(self, index):
        if self.single:
            [array] = self.arrays
            return array[index]
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]
