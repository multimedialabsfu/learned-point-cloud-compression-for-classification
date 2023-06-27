from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from compressai_trainer.run.train import _main

thisdir = Path(__file__).parent
config_path = thisdir.joinpath("../../conf")


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    _main(conf)


if __name__ == "__main__":
    main()
