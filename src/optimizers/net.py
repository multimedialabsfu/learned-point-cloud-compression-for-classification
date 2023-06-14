from __future__ import annotations

from typing import Any, Dict, Mapping, cast

import torch.nn as nn
import torch.optim as optim

from compressai.registry import OPTIMIZERS, register_optimizer


@register_optimizer("net")
def net_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns optimizer for net loss."""
    parameters = {
        "net": {name for name, param in net.named_parameters() if param.requires_grad},
    }

    params_dict = dict(net.named_parameters())

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net"]}

    return cast(Dict[str, optim.Optimizer], optimizer)
