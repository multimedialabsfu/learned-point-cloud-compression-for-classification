import compressai
import compressai_trainer
import compressai_trainer.config.env as _M
import src
from compressai_trainer.config.env import *

_M.PACKAGES = [compressai, compressai_trainer, src]
# _M.PACKAGES.append(src)
