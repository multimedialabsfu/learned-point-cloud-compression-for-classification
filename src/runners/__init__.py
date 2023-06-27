# from compressai_trainer.runners import *

from . import base as base
from . import utils as utils
from .classification_point_cloud import PointCloudClassificationRunner
from .multitask_point_cloud_compression import MultitaskPointCloudCompressionRunner
from .point_cloud_compression import PointCloudCompressionRunner

__all__ = [
    "MultitaskPointCloudCompressionRunner",
    "PointCloudClassificationRunner",
    "PointCloudCompressionRunner",
]
