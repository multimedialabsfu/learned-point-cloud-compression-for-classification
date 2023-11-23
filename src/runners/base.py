import compressai
import compressai_trainer
import compressai_trainer.runners.base as _M
import src
from compressai_trainer.runners.base import *

_M.PACKAGES = [compressai, compressai_trainer, src]
