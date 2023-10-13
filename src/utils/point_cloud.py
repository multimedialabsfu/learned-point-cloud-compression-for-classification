from __future__ import annotations

import os
import re
import subprocess
import traceback
import warnings
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

try:
    from memory_tempfile import MemoryTempfile

    tempfile = MemoryTempfile()
except ImportError:
    import tempfile

    warnings.warn("Could not import memory_tempfile. Falling back to tempfile.")
except Exception as e:
    import tempfile

    warnings.warn(
        "Could not create MemoryTempfile. Falling back to tempfile. "
        "Without the /tmp directory, this may lead to slower pc_error metrics. "
        "If you have a tmpfs at a different location, consider specifying it: "
        "tempfile = MemoryTempfile(['/path/to/tmpdir']). "
        "Otherwise, you can ignore this warning, since the default works too. "
        "Exception message:\n"
        f"{''.join(traceback.format_exception(None, e, e.__traceback__))}"
    )


def pc_write(pc: np.ndarray, file):
    assert pc.ndim == 2 and pc.shape[1] == 3
    vertex = np.array(
        list(zip(pc[:, 0], pc[:, 1], pc[:, 2])),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    data = PlyData([PlyElement.describe(vertex, "vertex")])
    data.write(file)


def pc_read(file) -> np.ndarray:
    data = PlyData.read(file)
    vertex = data["vertex"].data
    pc = np.stack((vertex["x"], vertex["y"], vertex["z"]))
    return np.ascontiguousarray(pc.T)


def pc_error_run(
    x: np.ndarray,
    x_hat: np.ndarray,
    peak_value: Optional[float] = None,
    normals: Optional[torch.Tensor] = None,
    hausdorff: bool = True,
    color: bool = False,
    lidar: bool = False,
    single_pass: bool = False,
) -> str:
    if normals is not None:
        raise NotImplementedError

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "x.ply"), "wb") as f:
            pc_write(x, f)

        with open(os.path.join(tmpdir, "x_hat.ply"), "wb") as f:
            pc_write(x_hat, f)

        cmd = ["pc_error", "-a", "x.ply", "-b", "x_hat.ply"]
        cmd += ["--resolution", str(peak_value)] if peak_value is not None else []
        cmd += ["--hausdorff"] if hausdorff else []
        cmd += ["--color"] if color else []
        cmd += ["--lidar"] if lidar else []
        cmd += ["--singlePass"] if single_pass else []

        process = subprocess.run(cmd, cwd=tmpdir, capture_output=True)

    return process.stdout.decode("utf-8")


def pc_error_parse_output(output: str) -> dict:
    results = {}
    mode = "start"

    for line in output.splitlines():
        if "Final (symmetric)" in line:
            mode = "symmetric"
            continue

        if mode != "symmetric":
            continue

        pattern = r"\s*(?P<key>\S+)\s+\S*\((?P<type>\S+)\):\s+(?P<value>.+)"
        m = re.match(pattern, line)
        if not m:
            continue
        d = m.groupdict()

        if d["key"] == "mseF,PSNR" and d["type"] == "p2point":
            results["d1-psnr"] = float(d["value"])
        if d["key"] == "mseF,PSNR" and d["type"] == "p2plane":
            results["d2-psnr"] = float(d["value"])
        if d["key"] == "h.,PSNR" and d["type"] == "p2point":
            results["d1-psnr-hausdorff"] = float(d["value"])
        if d["key"] == "h.,PSNR" and d["type"] == "p2plane":
            results["d2-psnr-hausdorff"] = float(d["value"])

    return results
