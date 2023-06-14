# from compressai.losses import *

from .point_cloud_compression import (
    ChamferPccRateDistortionLoss,
    OrderedPccRateDistortionLoss,
)

__all__ = [
    "ChamferPccRateDistortionLoss",
    "OrderedPccRateDistortionLoss",
]
