import os
import subprocess
import tempfile
import time
from contextlib import contextmanager, suppress

import numpy as np
import torch

from compressai_trainer.utils.point_cloud import pc_read, pc_write


@contextmanager
def tmp_fifo():
    """Context Manager for creating named pipes with temporary names."""
    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "fifo")
    os.mkfifo(filename)
    try:
        yield filename
    finally:
        os.unlink(filename)
        os.rmdir(tmpdir)


x = torch.randn(2048, 3).numpy()
x_hat = x + 0.1 * torch.randn(x.shape).numpy()

t = time.time()

with suppress(OSError):
    os.remove("x.ply")
with suppress(OSError):
    os.remove("x_hat.ply")
# os.mkfifo("x.ply")
# os.mkfifo("x_hat.ply")
time.sleep(1)

# cmd = ["pc_error", "-a", "x.ply", "-b", "x_hat.ply", "--hausdorff"]
cmd = ["cat", "x.ply", "x_hat.ply"]
# cmd = ["cat", "x.ply"]
process = subprocess.Popen(cmd)

time.sleep(1)
print("opening...")
# with os.open("x.ply", os.O_WRONLY | os.O_NONBLOCK) as f:


def opener(path, flags):
    print(flags)
    return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    # return os.open(path, flags, dir_fd=dir_fd)


with (
    open("x.ply", "wb") as f,
    # open("x.ply", "wb", opener=opener) as f,
):
    pc_write(x, f)

time.sleep(1)

with (
    open("x_hat.ply", "wb") as f_hat,
    # open("x_hat.ply", "wb", opener=opener) as f_hat,
):
    pc_write(x_hat, f_hat)

# with (
#     open("x.ply", "wb", opener=opener) as f,
#     open("x_hat.ply", "wb", opener=opener) as f_hat,
# ):
#     pc_write(x, f)
#     pc_write(x_hat, f_hat)


process.wait()

# with open("x.ply", "wb") as f, open("x.ply", "wb") as f_hat:
# with tmp_fifo() as x_filename, tmp_fifo() as x_hat_filename:
#     # pc_write(x, "x.ply")
#     # pc_write(x_hat, "x_hat.ply")
#
#     with open(x_filename, "wb") as f:
#         f.write("Hello\n")
#
#     # with open(x_filename, "wb") as f, open(x_hat_filename, "wb") as f_hat:
#     #     pc_write(x, f)
#     #     pc_write(x_hat, f_hat)

# print("run pc_error")
# cmd = ["pc_error", "-a", "x.ply", "-b", "x_hat.ply", "--hausdorff"]
# subprocess.run(cmd)

dt = time.time() - t

print(dt)
print(x)

x_ = pc_read("x.ply")
x_hat_ = pc_read("x_hat.ply")

assert np.all(np.isclose(x, x_))
assert np.all(np.isclose(x_hat, x_hat_))
