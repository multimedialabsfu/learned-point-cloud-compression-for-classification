from __future__ import annotations

import torch.nn as nn

from src.layers.ulhaqm import NamedLayer


class BaseClassificationPcModel(nn.Module):
    def _setup_hooks(self):
        def hook(module, input, output):
            self.outputs[module.name] = output

        for _, module in self.task_backend.named_modules():
            if not isinstance(module, NamedLayer):
                continue
            module.register_forward_hook(hook)

    def forward(self, input):
        self.outputs = {}
        x = input["points"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        t_hat = self.task_backend(y)

        return {
            "t_hat": t_hat,
            **{k: v for k, v in self.outputs.items()},
            # Additional outputs:
            "y": y,
            "debug_outputs": {},
        }
