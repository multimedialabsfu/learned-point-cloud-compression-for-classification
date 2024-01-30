# Copied from https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/data/dataset.py
# BSD-style license: https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE

from typing import Dict, Tuple, TypeVar, Union

from torch.utils.data import Dataset

__all__ = [
    "StackDataset",
]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
T_dict = Dict[str, T_co]
T_tuple = Tuple[T_co, ...]
T_stack = TypeVar("T_stack", T_tuple, T_dict)


class StackDataset(Dataset[T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: Union[tuple, dict]

    def __init__(self, *args: Dataset[T_co], **kwargs: Dataset[T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in args):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in tmp):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
