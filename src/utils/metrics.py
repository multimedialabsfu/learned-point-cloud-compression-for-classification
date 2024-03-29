from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pytorch3d.loss
import torch
from torch.utils.data.dataloader import default_collate

import compressai_trainer.utils.metrics as _M

from .point_cloud import pc_error_parse_output, pc_error_run


def compute_metrics(
    x: torch.Tensor, x_hat: torch.Tensor, metrics: list[str]
) -> dict[str, float]:
    visited_funcs = set()
    out = {}
    for metric in metrics:
        func = _M._METRICS[metric]
        if func in visited_funcs:
            continue
        visited_funcs.add(func)
        result = func(x, x_hat)
        if isinstance(result, dict):
            out.update(result)
        else:
            out[metric] = result
    return out


# Monkey patch:
_M.compute_metrics = compute_metrics


def chamfer_distance(a, b, **kwargs):
    a = a["pos"]
    b = b["x_hat"]
    loss, _ = pytorch3d.loss.chamfer_distance(a, b, **kwargs)
    return loss.item()


def pc_error(
    a,
    b,
    # a: torch.Tensor,
    # b: torch.Tensor,
    peak_value: Optional[float] = None,
    normals: Optional[torch.Tensor] = None,
    hausdorff: bool = True,
    color: bool = False,
    lidar: bool = False,
    single_pass: bool = False,
):
    """Point cloud error metrics using MPEG's pc_error tool."""
    a = a["pos"]
    b = b["x_hat"]

    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()

    def f(a, b):
        output = pc_error_run(
            a,
            b,
            peak_value=peak_value,
            normals=normals,
            hausdorff=hausdorff,
            color=color,
            lidar=lidar,
            single_pass=single_pass,
        )
        return pc_error_parse_output(output)

    return _vectorize(a, b, f)


def pc_acc_topk(target, output):
    outputs = output["t_hat"]
    labels = target["label"].squeeze(1)
    n = labels.shape[0]
    _, predicted = torch.max(outputs, dim=1)
    _, predicted_top3 = torch.topk(outputs, k=3, dim=1)
    assert predicted.shape == labels.shape
    return {
        "acc_top1": (predicted == labels).sum().item() / n,
        "acc_top3": (predicted_top3 == labels[..., None]).sum().item() / n,
    }


def _vectorize(a, b, f, num_workers=4):
    num_workers = min(num_workers, len(a))
    if num_workers == 1:
        ds = [f(a[i], b[i]) for i in range(len(a))]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            ds = list(executor.map(f, a, b))
    d = default_collate(ds)
    return {k: v.mean().item() for k, v in d.items()}


_M._METRICS = {
    **_M._METRICS,
    "acc_top1": pc_acc_topk,
    "acc_top3": pc_acc_topk,
    "chamfer": chamfer_distance,
    "d1-psnr": pc_error,
    "d2-psnr": pc_error,
    "d1-psnr-hausdorff": pc_error,
    "d2-psnr-hausdorff": pc_error,
    "pc_error": pc_error,
    "pc_acc_topk": pc_acc_topk,
    # "pc_acc_top1": lambda *args: pc_acc_topk(*args, k=1),
    # "pc_acc_top3": lambda *args: pc_acc_topk(*args, k=3),
}
