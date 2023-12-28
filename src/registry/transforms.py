from typing import Callable, Dict, Type, TypeVar

import torch_geometric.transforms
import torchvision.transforms

import compressai.registry as _M
from compressai.registry import TRANSFORMS
from src.typing import TTransform

__all__ = [
    "TRANSFORMS",
    "register_transform",
]

TRANSFORMS_NEW: Dict[str, Callable[..., TTransform]] = {
    **{k: v for k, v in torchvision.transforms.__dict__.items() if k[0].isupper()},
    **{k: v for k, v in torch_geometric.transforms.__dict__.items() if k[0].isupper()},
}

TRANSFORMS.update(TRANSFORMS_NEW)

TTransform_b = TypeVar("TTransform_b", bound=TTransform)


def register_transform(name: str):
    """Decorator for registering a transform."""

    def decorator(cls: Type[TTransform_b]) -> Type[TTransform_b]:
        TRANSFORMS[name] = cls
        return cls

    return decorator


# Monkey patch:
# _M.torchvision.TRANSFORMS.update(TRANSFORMS_NEW)
_M.register_transform = register_transform
