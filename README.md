# Learned Point Cloud Compression for Classification

> <sup>**Abstract:** Deep learning is increasingly being used to perform machine vision tasks such as classification, object detection, and segmentation on 3D point cloud data. However, deep learning inference is computationally expensive. The limited computational capabilities of end devices thus necessitate a codec for transmitting point cloud data over the network for server-side processing. Such a codec must be lightweight and capable of achieving high compression ratios without sacrificing accuracy. Motivated by this, we present a novel point cloud codec that is highly specialized for the machine task of classification. Our codec, based on PointNet, achieves a significantly better rate-accuracy trade-off in comparison to alternative methods. In particular, it achieves a 94% reduction in BD-bitrate over non-specialized codecs on the ModelNet40 dataset. For low-resource end devices, we also propose two lightweight configurations of our encoder that achieve similar BD-bitrate reductions of 93% and 92% with 3% and 5% drops in top-1 accuracy, while consuming only 0.470 and 0.048 encoder-side kMACs/point, respectively. Our codec demonstrates the potential of specialized codecs for machine analysis of point clouds, and provides a basis for extension to more complex tasks and datasets in the future.</sup>

- **Authors:** Mateen Ulhaq and Ivan V. Bajić
- **Affiliation:** Simon Fraser University
- **Paper:** Published at MMSP 2023. Available online: [[arXiv](TODO)]. [[BibTeX citation](#citation)]


----


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


## Saving results to JSON

```bash
python scripts/save_json_from_aim_query.py --aim-repo-path="/path/to/aim/repo" --output-dir="results/point-cloud-classification/modelnet40"
```


## Plotting

See `scripts/run_plot.sh` for examples.


## Citation

Please cite this work as:

```bibtex
@inproceedings{ulhaq2023mmsp,
  title = {Learned Point Cloud Compression for Classification},
  author = {Ulhaq, Mateen and Baji\'{c}, Ivan V.},
  booktitle = {Proc. IEEE MMSP},
  year = {2023},
}
```
