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
    runner.model_module.update(force=True)

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []

    for batch, filename in zip(batches, filenames):
        batch = {k: v.to(runner.engine.device) for k, v in batch.items()}
        out_infer = runner.predict_batch(batch)
        out_criterion = runner.criterion(out_infer["out_net"], batch)
        out_metrics = compute_metrics(batch, out_infer["out_dec"]["x_hat"], metrics)

        output = {
            "filename": filename,
            "bpp": out_infer["bpp"],
            **{
                k: v.item()
                for k, v in out_criterion.items()
                if k in runner.hparams["runner"]["meters"]["infer"]
            },
            **out_metrics,
            "encoding_time": out_infer["encoding_time"],
            "decoding_time": out_infer["decoding_time"],
        }
        outputs.append(output)
        print(json.dumps(output, indent=2))

        output_filename = (output_dir / filename).with_suffix("")
        os.makedirs(output_filename.parent, exist_ok=True)
        _write_bitstreams(out_infer["out_enc"]["strings"], output_filename)
        if "x_hat" in out_infer["out_dec"]:
            _write_pointcloud(out_infer["out_dec"]["x_hat"], output_filename)
        if "t_hat" in out_infer["out_dec"]:
            t_hat_probs = out_infer["out_dec"]["t_hat"].softmax(1)
            _write_probs(t_hat_probs, output_filename)

    return outputs


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


if __name__ == "__main__":
    _main()
