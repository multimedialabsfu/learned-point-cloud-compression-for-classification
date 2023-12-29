from __future__ import annotations

from typing import Mapping

from compressai.latent_codecs import LatentCodec
from compressai.models import CompressionModel
from src.layers import NamedLayer


class BaseClassificationPccModel(CompressionModel):
    latent_codec: Mapping[str, LatentCodec]

    def _setup_hooks(self):
        def hook(module, input, output):
            self.outputs[module.name] = output

        for _, module in self.task_backend.named_modules():
            if not isinstance(module, NamedLayer):
                continue
            module.register_forward_hook(hook)

    def forward(self, input):
        self.outputs = {}
        x = input["pos"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec["y"](y)
        y_hat = y_out["y_hat"]
        t_hat = self.task_backend(y_hat)

        return {
            "t_hat": t_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            **{k: v for k, v in self.outputs.items()},
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        self.outputs = {}
        x = input["pos"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec["y"].compress(y)
        [y_strings] = y_out["strings"]
        # NOTE: the shape in y_out["shape"] is incorrectly y.shape[-2:]
        return {"strings": [y_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        self.outputs = {}
        [y_strings] = strings
        y_out = self.latent_codec["y"].decompress([y_strings], shape)
        y_hat = y_out["y_hat"]
        t_hat = self.task_backend(y_hat)
        return {"t_hat": t_hat}
