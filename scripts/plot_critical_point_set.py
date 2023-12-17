import catalyst
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

import src.models  # noqa: F401
from compressai_trainer.run.eval_model import config_path, setup
from compressai_trainer.utils.metrics import compute_metrics


def test_critical_point_set(g_a_1, g_a_2, x):
    ga1_x = g_a_1(x.transpose(-1, -2))
    y = g_a_2(ga1_x)

    idx_max = ga1_x.argmax(axis=-1)
    x_max = x[0, idx_max]

    ga1_x_max = g_a_1(x_max.transpose(-1, -2))
    y_max = g_a_2(ga1_x_max)

    # Demonstrate that x_max was indeed a critical point set.
    assert torch.isclose(y, y_max).all()


def write_mpl_figure(path, df):
    # Defaults to (6.4, 4.8)  -->  (4.8, 4.8) for square aspect ratio.
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(projection="3d")

    if "critical" not in df.columns:
        df["critical"] = False

    for is_critical, df_ in df.groupby("critical", sort=True):
        ax.scatter(
            df_["x"],
            df_["y"],
            df_["z"],
            marker="+" if is_critical else ".",
            s=1024.0 if is_critical else 64.0,
            color="tab:red" if is_critical else "tab:blue",
        )

    set_mpl_camera_scale(ax, scale=1.05)
    ax.view_init(elev=45, azim=-145, roll=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    # fig.tight_layout()
    # fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def set_mpl_camera_scale(ax, scale=1):
    ax.set_xlim3d(-scale / 2, scale / 2)
    ax.set_ylim3d(-scale / 2, scale / 2)
    ax.set_zlim3d(-scale / 2, scale / 2)


@torch.no_grad()
@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    # Remove train and valid datasets.
    dataset = dict(conf["dataset"])
    del dataset["train"]
    del dataset["valid"]
    conf.dataset = dataset

    # Setup runner from hydra config.
    runner = setup(conf)
    runner.model_module.update(force=True)
    device = runner.engine.device
    model = runner.model
    batches = runner.loaders["infer"]

    # Force re-seed to ensure consistent dataset.
    catalyst.utils.set_global_seed(conf.misc.seed)

    # Pick sample from dataset.
    batch = next(iter(batches))
    batch = {k: v[None, 3].to(device) for k, v in batch.items()}
    x = batch["points"]

    is_pointnet = conf.model.name.endswith("pointnet")

    if is_pointnet:
        # Split g_a into pre- and post-pool.
        g_a_1 = model.g_a[:-2]
        g_a_2 = model.g_a[-2:]
        print(g_a_1)
        print(g_a_2)
        assert (
            isinstance(g_a_2[0], torch.nn.AdaptiveMaxPool1d)
            and g_a_2[0].output_size == 1
        )

        # Verify g_a_1(S) = g_a_1(S'), where S' is a critical point subset of S.
        x = 2 * torch.rand((1, 1024, 3), device=device) - 1
        test_critical_point_set(g_a_1, g_a_2, x)

        # Compute critical point set indices.
        assert x.shape[0] == 1
        ga1_x = g_a_1(x.transpose(-1, -2))
        idx_max = ga1_x.argmax(axis=-1)
        # x_max = x[0, idx_max]

    # Write critical point set figure.
    df = pd.DataFrame(x[0].cpu().numpy(), columns=["x", "y", "z"])
    if is_pointnet:
        critical = np.zeros_like(df["x"], dtype=bool)
        critical[idx_max[0].cpu().numpy()] = True
        df["critical"] = critical
    else:
        df["critical"] = False
    write_mpl_figure(conf.misc.out_path.critical, df)

    # Write reconstruction figure.
    out_net = model({"points": x})
    df = pd.DataFrame(out_net["x_hat"][0].cpu().numpy(), columns=["x", "y", "z"])
    write_mpl_figure(conf.misc.out_path.reconstruction, df)

    # Compute metrics.
    out_infer = runner.predict_batch(batch, **runner._inference_kwargs)
    out_metrics = compute_metrics(
        batch, out_infer["out_net"], ["chamfer", "pc_error", "pc_acc_topk"]
    )
    correct = out_metrics["acc_top1"]
    final_metrics = {
        "bits": out_infer["bpp"],
        **out_metrics,
    }
    print(final_metrics)
    print(out_infer["out_net"]["t_hat"].argmax(axis=-1))
    print(
        f"% {final_metrics['bits']:.0f} bits, {final_metrics['d1-psnr']:.2f} D1-PSNR, {correct}"
    )

    # x = 2 * torch.rand((1, 1024, 3), device=device) - 1
    # ga1_x = g_a_1(x.transpose(-1, -2))
    # idx = ga1_x.argsort(axis=-1)
    # TODO plot intensity=ga1_x, color=x[idx][..., 2, :]  # z axis...?
    # df = ...


if __name__ == "__main__":
    main()
