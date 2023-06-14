from __future__ import annotations

import torch

import compressai.entropy_models.entropy_models as _M


# NOTE: This is a copy of the original update() method,
# with the `self._update_quantiles()` call added.
def update(self, force: bool = False) -> bool:
    # Check if we need to update the bottleneck parameters, the offsets are
    # only computed and stored when the conditonal model is update()'d.
    if self._offset.numel() > 0 and not force:
        return False

    # NOTE: This line is not in the original method:
    self._update_quantiles()

    medians = self.quantiles[:, 0, 1]

    minima = medians - self.quantiles[:, 0, 0]
    minima = torch.ceil(minima).int()
    minima = torch.clamp(minima, min=0)

    maxima = self.quantiles[:, 0, 2] - medians
    maxima = torch.ceil(maxima).int()
    maxima = torch.clamp(maxima, min=0)

    self._offset = -minima

    pmf_start = medians - minima
    pmf_length = maxima + minima + 1

    max_length = pmf_length.max().item()
    device = pmf_start.device
    samples = torch.arange(max_length, device=device)
    samples = samples[None, :] + pmf_start[:, None, None]

    pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
    pmf = pmf[:, 0, :]
    tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

    quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
    self._quantized_cdf = quantized_cdf
    self._cdf_length = pmf_length + 2
    return True


@torch.no_grad()
def _update_quantiles(self):
    device = self.quantiles.device
    shape = (self.channels, 1, 1)
    low = torch.full(shape, -1e9, device=device)
    high = torch.full(shape, 1e9, device=device)

    def f(y, self=self):
        return self._logits_cumulative(y, stop_gradient=True)

    for i in range(len(self.target)):
        q_i = self._search_target(f, self.target[i], low, high)
        self.quantiles[:, :, i] = q_i[:, :, 0]


@staticmethod
def _search_target(f, target, low, high):
    assert (low <= high).all()
    assert ((f(low) <= target) & (target <= f(high))).all()
    while not torch.isclose(low, high).all():
        mid = (low + high) / 2
        f_mid = f(mid)
        low = torch.where(f_mid <= target, mid, low)
        high = torch.where(f_mid >= target, mid, high)
    return (low + high) / 2


# Monkey patch so that update() also calls _update_quantiles().
# This is an alternative to minimizing aux_loss.
# See also: https://github.com/InterDigitalInc/CompressAI/pull/231
#
_M.EntropyBottleneck.update = update
_M.EntropyBottleneck._update_quantiles = _update_quantiles
_M.EntropyBottleneck._search_target = _search_target
