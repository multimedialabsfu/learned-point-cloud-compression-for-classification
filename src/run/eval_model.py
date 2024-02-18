from __future__ import annotations

import json
import os
from pathlib import Path

import compressai_trainer.run.eval_model as _M
from compressai_trainer.run.eval_model import *
from compressai_trainer.run.eval_model import _write_bitstreams
from compressai_trainer.run.eval_model import main as _main
from src.utils.metrics import compute_metrics
from src.utils.point_cloud import pc_write

_M.thisdir = Path(__file__).parent
_M.config_path = _M.thisdir.joinpath("../../conf")


def run_eval_model(runner, batches, filenames, output_dir, metrics):
    if hasattr(runner.model_module, "update"):
        runner.model_module.update(force=True)

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []

    for batch, filename in zip(batches, filenames):
        batch = {k: v.to(runner.engine.device) for k, v in batch.items()}
        out_infer = runner.predict_batch(batch)
        out_criterion = runner.criterion(out_infer["out_net"], batch)
        out_hat = {**out_infer["out_net"], **out_infer.get("out_dec", {})}
        out_metrics = compute_metrics(batch, out_hat, metrics)

        output = {
            "filename": filename,
            "bpp": out_infer.get("bpp", np.nan),
            **{
                k: v.item()
                for k, v in out_criterion.items()
                if k in runner.hparams["runner"]["meters"]["infer"]
            },
            **out_metrics,
            "encoding_time": out_infer.get("encoding_time", np.nan),
            "decoding_time": out_infer.get("decoding_time", np.nan),
        }
        outputs.append(output)
        print(json.dumps(output, indent=2))

        output_filename = (output_dir / filename).with_suffix("")
        os.makedirs(output_filename.parent, exist_ok=True)

        # Write bitstreams.
        if "out_enc" not in out_infer:
            pass
        elif isinstance(out_infer["out_enc"]["strings"], list):
            _write_bitstreams(out_infer["out_enc"]["strings"], output_filename)
        elif isinstance(out_infer["out_enc"]["strings"], dict):
            # HACK: very ad-hoc
            # [[[a1, a2], [b1, b2]], [[c1, c2], [d1, d2], [e1, e2]]]
            #   -> [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]]
            _write_bitstreams(
                [xs for xss in out_infer["out_enc"]["strings"].values() for xs in xss],
                output_filename,
            )

        if "out_dec" not in out_infer:
            continue
        if "x_hat" in out_infer["out_dec"]:
            _write_pointcloud(out_infer["out_dec"]["x_hat"], output_filename)
        if "t_hat" in out_infer["out_dec"]:
            t_hat_probs = out_infer["out_dec"]["t_hat"].softmax(1)
            _write_probs(t_hat_probs, output_filename)

    return outputs


def _results_dict(conf, outputs):
    result_keys = list(outputs[0].keys())
    result_non_avg_keys = ["filename"]
    result_avg_keys = [k for k in result_keys if k not in result_non_avg_keys]

    return {
        "name": conf.model.name,
        "description": "",
        "meta": {
            "dataset": conf.dataset.infer.meta.name,
            "env.aim.run_hash": conf.env.aim.run_hash,
            "misc.device": conf.misc.device,
            "model.source": conf.model.get("source"),
            "model.name": conf.model.get("name"),
            "model.metric": conf.model.get("metric"),
            "model.quality": conf.model.get("quality"),
            **_flatten_dict_to_pathstr_dict(conf.criterion.lmbda, "criterion.lmbda"),
            "paths.model_checkpoint": conf.paths.get("model_checkpoint"),
        },
        "results_averaged": {
            **{k: np.mean([out[k] for out in outputs]) for k in result_avg_keys},
        },
        "results_by_sample": {
            **{k: [out[k] for out in outputs] for k in result_keys},
        },
    }


def _flatten_dict_to_pathstr_dict(d, prefix="", sep="."):
    """
    Examples:
        >>> _flatten_dict_to_pathstr_dict({"cls": 1, "fm": {"a": 2}})
        {'cls': 1, 'fm.a': 2}
    """
    if not hasattr(d, "items"):
        return {prefix: d}
    return {
        k: v
        for kk, vv in d.items()
        for k, v in _flatten_dict_to_pathstr_dict(
            vv, prefix=f"{prefix}{sep}{kk}" if prefix else kk, sep=sep
        ).items()
    }


def _write_pointcloud(x, filename):
    B, _, C = x.shape
    assert B == 1 and C == 3
    with open(filename.with_suffix(".ply"), "wb") as f:
        pc_write(x.squeeze(0).detach().cpu().numpy(), f)


def _write_probs(x, filename):
    B, T = x.shape
    assert B == 1
    x = x.squeeze(0).detach().cpu().numpy()
    with open(filename.with_suffix(".probs"), "w") as f:
        for i in range(T):
            print(x[i], file=f)


def main():
    _main()


# Monkey patch:
_M.run_eval_model = run_eval_model
_M._results_dict = _results_dict


if __name__ == "__main__":
    _main()
