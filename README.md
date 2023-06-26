## Installation

### Using poetry

```bash
poetry install
poetry shell
pip install -e ./submodules/compressai
pip install -e ./submodules/compressai-trainer
pip install torch==1.13.1 torchvision==0.14.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

TODO: compressai-trainer dependency relax torch



### `pc_error` tool

```bash
cd third_party/pc_error
make

# Install for user.
mkdir -p "$HOME/.local/bin/"
cp pc_error "$HOME/.local/bin/"
```


### tmc13 codec

```bash
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13
cd mpeg-pcc-tmc13
mkdir -p build
cd build
cmake ..
make

# Install for user.
mkdir -p "$HOME/.local/bin/"
cp tmc3/tmc3 "$HOME/.local/bin/"
```


### PATH

Ensure `$PATH` is set correctly:

```bash
echo "$PATH" | sed 's/:/\n/g' | grep -q "$HOME/.local/bin" || (
  echo "Please add $HOME/.local/bin to PATH"
  echo "For example, try running:"
  echo 'echo '"'"'export PATH="$PATH:$HOME/.local/bin/"'"'"' >> "$HOME/.bashrc"'
)
```



## TODO

- Example run command, e.g.

```python
poetry run python -m src.run.train --config-path="$PWD/conf/" --config-name="example_pcc_singletask"

poetry run compressai-train --config-path="$PWD/conf/" --config-name="example_pcc_singletask"   ++hp.num_points=1024  # ...
```

- Run commands for reproducing same models...


- Ensure everything installs/downloads/builds/trains/evaluates (100% reproducibility)
- Set latest versions of submodules
- Add Dockerfile
- Add dataset downloading/parsing scripts

For clarity:

- Documentation of usage
- Remove unused models, etc (e.g. pc_reconstruction), "ClusterAttention", losses, etc; perhaps in a separate "cleanup" commit for documentation

Separate commits:

- Rename "early split" or something like that? (Or is the other network a subset?)
- Rename mini-001 -> micro (model name, etc)
- Use same config structure for `g_a.transform.pointwise`
- `rate_format = "bpp|bit"`

