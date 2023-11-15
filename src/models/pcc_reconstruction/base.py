from __future__ import annotations

from compressai.latent_codecs import LatentCodec
from compressai.models import CompressionModel


class BaseReconstructionPccModel(CompressionModel):
    latent_codec: LatentCodec

    def forward(self, input):
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        assert x_hat.shape == x.shape

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec.compress(y)
        [y_strings] = y_out["strings"]
        return {"strings": [y_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        [y_strings] = strings
        y_hat = self.latent_codec.decompress([y_strings], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
