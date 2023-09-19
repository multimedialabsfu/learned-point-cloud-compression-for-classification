import sys

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from omegaconf import OmegaConf

from compressai.entropy_models import EntropyBottleneck
from compressai_trainer.run.eval_model import setup

run_hash = "5fc2276b120e427ebacf840d"
run_hash = "4796f2b0abb24c14a66ad77e"
run_hash = "b2edb7cc8fc540f8b547ea4b"  # ModelNet10
run_hash = "2fc03792a0a346afa56111dc"  # ModelNet40

run_hash = sys.argv[1]


def load_by_run_hash(run_hash):
    path = f"/home/mulhaq/data/runs/pc-mordor/{run_hash}/configs/config.yaml"
    conf = OmegaConf.load(path)
    del conf.dataset["train"]
    del conf.dataset["valid"]
    conf.model.source = "config"
    conf.paths.model_checkpoint = "${paths.checkpoints}/runner.last.pth"
    runner = setup(conf)
    return runner


@torch.no_grad()
def _update_quantiles(self):
    device = self.quantiles.device
    shape = (self.channels, 1, 1)
    low = torch.full(shape, -1e9, device=device)
    high = torch.full(shape, 1e9, device=device)

    def f(y, self=self):
        return self._logits_cumulative(y, stop_gradient=True)

    for i in range(len(self.target)):
        q_i = _search_target(f, self.target[i], low, high)
        self.quantiles[:, :, i] = q_i[:, :, 0]


def _search_target(f, target, low, high):
    assert (low <= high).all()
    assert ((f(low) <= target) & (target <= f(high))).all()
    while not torch.isclose(low, high).all():
        mid = (low + high) / 2
        f_mid = f(mid)
        low = torch.where(f_mid <= target, mid, low)
        high = torch.where(f_mid >= target, mid, high)
    return (low + high) / 2


def enc_dec_accuracy(runner):
    out_encs = []
    correct = []

    for batch in runner.loaders["infer"]:
        out_infer = runner.predict_batch(batch)
        out_encs.append(out_infer["out_enc"])

    for batch, out_enc in zip(runner.loaders["infer"], out_encs):
        out_dec = runner.model.decompress(out_enc["strings"], out_enc["shape"])
        pred_labels = out_dec["t_hat"].argmax(axis=-1).detach().cpu()
        correct.extend((batch["labels"] == pred_labels).numpy().tolist())

    correct = np.array(correct)
    lengths = np.array(
        [len(s) for out_enc in out_encs for ss in out_enc["strings"] for s in ss]
    )
    num_points = batch["points"].shape[1]
    bpp = lengths.mean() * 8 / num_points

    print(f"Num points: {num_points}")
    print(f"Top-1 accuracy: {correct.mean():.1%}")
    print(f"bpp: {bpp:.5f}")
    breakpoint()
    print("")


def plot_pdfs(entropy_bottleneck):
    c = entropy_bottleneck.channels
    q = entropy_bottleneck.quantiles[:, 0, :].detach()

    # Symmetrically supported pdfs:
    # offsets = torch.maximum(q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]).ceil().int()
    # sizes = 2 * offsets + 1

    # left/right == "minima/maxima"
    left = (q[:, 1] - q[:, 0]).ceil().int()
    right = (q[:, 2] - q[:, 1]).ceil().int()
    offsets = left
    sizes = left + 1 + right

    max_size = sizes.max().item()
    num_samples = 2**0 * (max_size - 1) + 1

    # t = torch.arange(max_size, dtype=torch.float32, device="cuda")
    t = torch.linspace(0, max_size - 1, num_samples, device="cuda")
    dy = t[None, :] - offsets[:, None]
    y = dy + q[:, 1, None]

    with torch.no_grad():
        y_hat, y_likelihoods = entropy_bottleneck(y.unsqueeze(0), training=False)

    def trim(y):
        assert y.shape == (c, num_samples)
        # return y.cpu().numpy().reshape(-1)
        y = y.cpu().numpy()
        xss = [y[i, :l] for i, l in enumerate(sizes.cpu().tolist())]
        return [x for xs in xss for x in xs]

    # TODO Also, scale cdf_max to [0, 1] and multiply by 2**16 or something distribution length-preserving...
    # TODO also, deal with tail mass probabilities... normalize?
    # TODO Also, assert cdf[:, 0] == 0
    # Also, what is the bypass symbol?

    # TODO Also, it's quite weird that cdfs are 0 after cdf_sizes ends...

    pdfs = y_likelihoods.squeeze(0)
    cdfs = y_likelihoods.squeeze(0).cumsum(-1)
    medians = q[:, 1, None].repeat(1, num_samples)
    offsets = offsets[:, None].repeat(1, num_samples)

    name = [f"{i}" for i in range(c) for _ in range(sizes[i].item())]

    # TODO check that our cdfs match the generated ones...
    # also, is offset same sign as original code?
    # entropy_bottleneck.quantized_cdf
    # entropy_bottleneck.offset

    # TODO uhh why is entropy_bottleneck._quantized_cdf / 65536 first cdf so... linear?!
    # shouldn't it be a step function?
    # TODO maybe replace old cdf generation code with ours (i.e. repair...)

    # TODO bpp_loss is not the same as bpp... why? compare output y_likelihoods...

    d = {
        "name": name,
        "y": trim(y),
        "dy": trim(dy),
        "medians": trim(medians),
        "offsets": trim(offsets),
        "p_y": trim(pdfs),
        "cdfs": trim(cdfs),
    }

    print({k: len(v) for k, v in d.items()})

    df = pd.DataFrame.from_dict(d)

    fig = px.line(
        df,
        x="dy",
        y="p_y",
        color="name",
        hover_data=["y", "dy", "p_y", "medians", "offsets"],
        line_shape="hvh",
    )

    breakpoint()

    fig.show()


def print_stuff():
    x = next(iter(runner.loaders["infer"])).cuda()

    with torch.no_grad():
        out_infer = runner.predict_batch(x)

    y = out_infer["out_net"]["y"]
    y_hat = out_infer["out_net"]["y_hat"]
    y_likelihoods = out_infer["out_net"]["likelihoods"]["y"]
    x_hat = out_infer["out_net"]["x_hat"]

    x_hat_dec = out_infer["out_dec"]["x_hat"]

    print(y)
    print(y.shape)
    print(y.min(), y.max())

    num_points = np.prod(x.shape[:-1])
    bpp_loss = -torch.log2(y_likelihoods).sum() / num_points
    print(bpp_loss)

    bpp_losses = -torch.log2(y_likelihoods).sum(axis=1) / x.shape[1]
    print(bpp_losses.reshape(32))

    bpps = [len(s) * 8 / x.shape[1] for s in out_infer["out_enc"]["strings"][0]]
    print(bpps)

    # TODO still haven't compared "actual" cdfs to ours...

    # TODO compare each element's bpp / bpp_loss cost...
    # How to do this for bpp?

    # TODO isolate which channel(s) are causing problems
    # check their ACTUAL cdfs

    # TODO Perhaps compare what eb.compress(medians) is...
    # i.e. how many bits does it cost to compress y=medians

    # x_hat and x are not related...!
    # x_hat, x_hat_dec are equal
    # y, y_hat are very close


def get_living_channels(entropy_bottleneck, max_p=0.99):
    qcdf = entropy_bottleneck._quantized_cdf
    CDF_MAX = 2**16 - 1
    qpdf = (qcdf[..., 1:] - qcdf[..., :-1]).clip(0, CDF_MAX)
    dead = (qpdf / CDF_MAX >= max_p).any(axis=-1)
    alive = ~dead
    return alive


def main():
    runner = load_by_run_hash(run_hash)

    for _, m in runner.model.named_modules():
        if isinstance(m, EntropyBottleneck):
            _update_quantiles(m)

    # entropy_bottleneck = runner.model.entropy_bottleneck
    entropy_bottleneck = runner.model.latent_codec["y1"].entropy_bottleneck
    eb = entropy_bottleneck

    # gain = runner.model.g_a[-1].gain.reshape(-1).detach().cpu()
    # print(gain.argsort()[::-1].tolist())

    alive = get_living_channels(entropy_bottleneck)
    has_gain = hasattr(runner.model.g_a[-1], "gain")

    print(">>>")
    print(run_hash)
    print(f"Alive: {alive.sum()}")
    print(f"Has gain: {has_gain}")
    gain = runner.model.g_a[-1].gain.reshape(-1).detach().cpu()
    # print(gain.argsort(descending=True).tolist())
    print(gain.sort(descending=True)[0][: int(alive.sum().item())].tolist())
    print("")

    return

    breakpoint()

    enc_dec_accuracy(runner)

    plot_pdfs(entropy_bottleneck)


if __name__ == "__main__":
    main()


# Refactorings:
# - Rename "targets" to "target_masses" or similar
#
# Aux loss reasons:
# - Maybe for sharp cdfs, it is hard to locate a value of y that matches target likelihoods
#
# bpp vs bpp_loss mismatch reasons:
# - Maybe insufficient cdf sizes / bad offset computation? e.g. off by one error?
