- Ensure everything installs/downloads/builds/trains/evaluates (100% reproducibility)
- Add Dockerfile
- Add dataset downloading/parsing scripts

For clarity:

- Full documentation of usage
- Remove unused experimental models (e.g. pc_reconstruction), layers (e.g. ClusterAttention), losses
- Rename mini-001 --> micro to match with paper.
- Use same config structure for `g_a.transform.pointwise`
- `rate_format = "bpp|bit"` (reduce confusion when using bpp when it's actually bit)
