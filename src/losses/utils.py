def compute_rate_loss(likelihoods, batch_size, bit_per_bpp):
    out_bit = {
        f"bit_{name}_loss": lh.log2().sum() / -batch_size
        for name, lh in likelihoods.items()
    }
    out_bpp = {
        f"bpp_{name}_loss": out_bit[f"bit_{name}_loss"] / bit_per_bpp
        for name in likelihoods.keys()
    }
    out = {**out_bit, **out_bpp}
    out["bit_loss"] = sum(out_bit.values())
    out["bpp_loss"] = out["bit_loss"] / bit_per_bpp
    return out
