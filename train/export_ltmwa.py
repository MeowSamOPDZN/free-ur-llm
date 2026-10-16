"""Export checkpoints/ltmwa_final.pt -> ../ltmwa.json + ../ltmwa_weights.bin.

Matmul weights -> symmetric int8 per-tensor; biases + norms stay fp32.
The head is tied to tok_emb (no separate head_w stored).
"""
import json
import os
import struct

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def quant_i8(t):
    t = t.float()
    amax = t.abs().max().item() or 1e-8
    scale = amax / 127.0
    q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
    return q.numpy().tobytes(), scale


def main():
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "ltmwa_final.pt"),
                      map_location="cpu", weights_only=False)
    sd, cfg = ckpt["model"], ckpt["config"]
    n_layer = cfg["n_layer"]

    with open(os.path.join(HERE, "ltmwa_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)["tokens"]

    blob = bytearray()
    tensors = []

    def add_i8(name, t, shape):
        raw, scale = quant_i8(t)
        tensors.append({"name": name, "shape": shape, "dtype": "i8",
                        "scale": scale, "offset": len(blob)})
        blob.extend(raw)

    def add_f32(name, t, shape):
        raw = t.float().numpy().tobytes()
        tensors.append({"name": name, "shape": shape, "dtype": "f32",
                        "offset": len(blob)})
        blob.extend(raw)

    D = cfg["d_model"]
    add_i8("tok_emb", sd["tok_emb.weight"], [cfg["vocab_size"], D])
    add_i8("wpe", sd["wpe.weight"], [cfg["ctx"], D])
    for i in range(n_layer):
        p = f"blocks.{i}."
        add_f32(f"b{i}.ln1_w", sd[p + "ln1.weight"], [D])
        add_f32(f"b{i}.ln1_b", sd[p + "ln1.bias"], [D])
        add_i8(f"b{i}.qkv_w", sd[p + "attn.qkv.weight"], [3 * D, D])
        add_f32(f"b{i}.qkv_b", sd[p + "attn.qkv.bias"], [3 * D])
        add_i8(f"b{i}.proj_w", sd[p + "attn.proj.weight"], [D, D])
        add_f32(f"b{i}.proj_b", sd[p + "attn.proj.bias"], [D])
        add_f32(f"b{i}.ln2_w", sd[p + "ln2.weight"], [D])
        add_f32(f"b{i}.ln2_b", sd[p + "ln2.bias"], [D])
        add_i8(f"b{i}.mlp_fc_w", sd[p + "mlp_fc.weight"], [4 * D, D])
        add_f32(f"b{i}.mlp_fc_b", sd[p + "mlp_fc.bias"], [4 * D])
        add_i8(f"b{i}.mlp_out_w", sd[p + "mlp_out.weight"], [D, 4 * D])
        add_f32(f"b{i}.mlp_out_b", sd[p + "mlp_out.bias"], [D])
    add_f32("ln_f_w", sd["ln_f.weight"], [D])
    add_f32("ln_f_b", sd["ln_f.bias"], [D])
    add_f32("head_b", sd["head_b"], [cfg["vocab_size"]])

    with open(os.path.join(HERE, "..", "ltmwa_weights.bin"), "wb") as f:
        f.write(blob)
    meta = {"config": {"d_model": D, "n_layer": n_layer, "n_head": cfg["n_head"],
                       "ctx": cfg["ctx"], "vocab_size": cfg["vocab_size"],
                       "n_params": cfg["n_params"], "tied_head": True},
            "vocab": vocab,
            "tensors": tensors}
    with open(os.path.join(HERE, "..", "ltmwa.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"bin: {len(blob) / 1024 / 1024:.2f} MB, "
          f"meta: {os.path.getsize(os.path.join(HERE, '..', 'ltmwa.json')) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
