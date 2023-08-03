from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from compressai_trainer.config import create_dataloaders
from compressai_trainer.utils.point_cloud import pc_write

ROOTDIR = "by_n_ply"


@hydra.main(version_base=None, config_path="conf")
def main(conf: DictConfig):
    conf.dataset["train"].loader.shuffle = False
    loaders = create_dataloaders(conf)
    num_points = conf.hp.num_points

    loader_key = "infer"
    dataset_key = conf.dataset[loader_key].meta.name.lower()
    prev_label = 0
    idx = 0

    root = f"{ROOTDIR}/{num_points}/{dataset_key}"
    os.makedirs(root, exist_ok=True)

    for batch in loaders["infer"]:
        for label, points in zip(batch["labels"], batch["points"]):
            if prev_label != label:
                prev_label = label
                idx = 0
            path = f"{root}/{loader_key}_{label:02}_{idx:03}.ply"
            print(f"Writing {path}...")
            pc_write(points, path)
            idx += 1


if __name__ == "__main__":
    main()
