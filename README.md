TODO:

- Ensure everything installs/downloads/builds/trains/evaluates (100% reproducibility)

- Set correct versions of submodules
- Add results/*.json
- Add scripts/
- Add run.sh for training
- Add run_everything.sh... for "automatically" running everything!
- Add Dockerfile
- Add "wget datasets"
- Add config

For clarity:

- Documentation of usage
- Remove unused models, etc (e.g. pc_reconstruction), "ClusterAttention", losses, etc; perhaps in a separate "cleanup" commit for documentation

Separate commits:

- Rename "early split" or something like that? (Or is the other network a subset?)
- Rename mini-001 -> micro (model name, config, jsons, run.sh, etc)
- Use same config structure for `g_a.transform.pointwise`
- `rate_format = "bpp|bit"`


# Installation

TODO

```bash
poetry install
poetry shell
pip install -e ./submodules/compressai
pip install -e ./submodules/compressai-trainer
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```


bd rate
pc_error tool
tmc13
kaleido


# Generating datasets...

