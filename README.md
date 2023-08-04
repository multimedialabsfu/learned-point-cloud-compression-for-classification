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


## Training

Example run command for training a model:

```bash
python -m src.run.train --config-path="$PWD/conf/" --config-name="example_pcc_singletask" ++model.name="um-pcc-cls-only-pointnet-mmsp2023" ++criterion.lmbda.cls=100
```

Please see [`scripts/run.sh`](./scripts/run.sh) for more examples.


## Evaluation

CompressAI Trainer will evaluate trained models.

For other evaluation and analysis, see `scripts/`.


## Plotting

See `scripts/run_plot.sh` for examples.

