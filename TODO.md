- Reconstructed images/RD curves/etc
- Ensure everything installs/downloads/builds/trains/evaluates (100% reproducibility)
- Add Dockerfile
- Add dataset downloading/parsing scripts
- Upload pretrained models (that are compatible with "um-pcc-cls-only-pointnet")

For clarity:

- Full documentation of usage
- Use same config structure for `g_a.transform.pointwise`
- `rate_format = "bpp|bit"` (reduce confusion when using bpp when it's actually bit)
