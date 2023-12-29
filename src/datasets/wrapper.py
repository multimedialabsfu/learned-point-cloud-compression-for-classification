from torch.utils.data import Dataset
from torchvision import transforms

from compressai.registry import register_dataset
from compressai_trainer.config.dataset import create_data_transform
from src.registry import DATASETS


@register_dataset("WrapperDataset")
class WrapperDataset(Dataset):
    def __init__(self, *, _wrapper, **kwargs):
        for key in _wrapper.get("transform_keys", ()):
            kwargs[key] = transforms.Compose(
                [
                    create_data_transform(transform_conf)
                    for transform_conf in kwargs[key]
                ]
            )

        self.dataset = DATASETS[_wrapper["type"]](**kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __repr__(self):
        return f"WrapperDataset({self.dataset})"
