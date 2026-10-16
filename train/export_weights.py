"""Export checkpoints/final.pt -> ../weights.json for the browser engine (sml.js).

Keeps torch Linear layout: weight[out][in], y = Wx + b. Rounded to 5 decimals.
"""
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def r(t):
    return [[round(float(x), 5) for x in row] for row in t.tolist()]


def rv(t):
    return [round(float(x), 5) for x in t.tolist()]


def main():
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "final.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    cfg = ckpt["config"]
    n_layer = cfg["n_layer"]

    blocks = []
    for i in range(n_layer):
        p = f"blocks.{i}."
        blocks.append({
            "ln1_w": rv(sd[p + "ln1.weight"]),
            "ln1_b": rv(sd[p + "ln1.bias"]),
            "qkv_w": r(sd[p + "attn.qkv.weight"]),
            "qkv_b": rv(sd[p + "attn.qkv.bias"]),
            "proj_w": r(sd[p + "attn.proj.weight"]),
            "proj_b": rv(sd[p + "attn.proj.bias"]),
            "ln2_w": rv(sd[p + "ln2.weight"]),
            "ln2_b": rv(sd[p + "ln2.bias"]),
            "mlp_fc_w": r(sd[p + "mlp_fc.weight"]),
            "mlp_fc_b": rv(sd[p + "mlp_fc.bias"]),
            "mlp_out_w": r(sd[p + "mlp_out.weight"]),
            "mlp_out_b": rv(sd[p + "mlp_out.bias"]),
        })

    out = {
        "config": {
            "d_model": cfg["d_model"],
            "n_layer": n_layer,
            "n_head": cfg["n_head"],
            "ctx": cfg["ctx"],
            "vocab": cfg["vocab"],
            "n_params": cfg["n_params"],
        },
        "weights": {
            "tok_emb": r(sd["tok_emb.weight"]),
            "wpe": r(sd["wpe.weight"]),
            "blocks": blocks,
            "ln_f_w": rv(sd["ln_f.weight"]),
            "ln_f_b": rv(sd["ln_f.bias"]),
            "head_w": r(sd["head.weight"]),
            "head_b": rv(sd["head.bias"]),
        },
    }
    dest = os.path.join(HERE, "..", "weights.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    size = os.path.getsize(dest)
    print(f"wrote {dest} ({size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
