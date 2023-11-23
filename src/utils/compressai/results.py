from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

import compressai_trainer.utils.compressai.results as _M
import src

_M.GENERIC_CODECS = []


def compressai_dataframe(
    codec_name: str,
    dataset: str = "image/kodak",
    opt_metric: str = "mse",
    device: str = "cuda",
    source: str = "compressai",
    generic_codecs: list[str] = _M.GENERIC_CODECS,
    base_path: Optional[str] = None,
    filename_format: Optional[str] = None,
    **filename_format_kwargs,
) -> pd.DataFrame:
    """Returns a dataframe containing the results of a given codec."""
    if base_path is None:
        base_path = f"{src.__path__[0]}/../results/{dataset}"

    if filename_format is not None:
        pass
    elif codec_name in generic_codecs:
        filename_format = "{codec_name}"
    elif source == "paper":
        filename_format = "{source}-{codec_name}"
    elif source == "compressai":
        filename_format = "{source}-{codec_name}_{opt_metric}_{device}_{name}"
    else:
        raise ValueError(f"Unknown source: {source}")

    filename = filename_format.format(
        source=source,
        codec_name=codec_name,
        opt_metric=opt_metric,
        device=device,
        **filename_format_kwargs,
    )

    with open(f"{base_path}/{filename}.json") as f:
        d = json.load(f)

    df = compressai_json_to_dataframe(d)
    return df


def compressai_json_to_dataframe(d: dict[str, Any]) -> pd.DataFrame:
    d["results"] = _process_results(_M._rename_results(d["results"]))
    df = pd.DataFrame.from_dict(d["results"])
    df["name"] = d.get("name")
    df["model.name"] = d.get("meta", {}).get("model.name")
    df["description"] = d.get("description")
    return df


def _process_results(results):
    if "ms-ssim" in results:
        # NOTE: The dB of the mean of MS-SSIM samples
        # is not the same as the mean of MS-SSIM dB samples.
        results["ms-ssim-db"] = (
            -10 * np.log10(1 - np.array(results["ms-ssim"]))
        ).tolist()
    return results


# Monkey patch
_M.compressai_dataframe = compressai_dataframe
_M.compressai_json_to_dataframe = compressai_json_to_dataframe
_M._process_results = _process_results
