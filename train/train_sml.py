"""Train FreeUr LLM (SML-1): a tiny char-level transformer from scratch on CPU.

Architecture (mirrored exactly by sml.js in the browser):
    tok_emb -> [ln1 -> causal MHA -> +res] -> [ln2 -> MLP(gelu) -> +res] x L
            -> ln_f -> linear head
No positional embeddings beyond learned ones (wpe).

Usage:
    ~/workspace/mnai-nano/venv/bin/python train_sml.py
Outputs: checkpoints/final.pt  (+ samples during training)
Then: export_weights.py -> ../weights.json
"""
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)

# ---- hyperparams ----
D_MODEL = 96
N_LAYER = 3
N_HEAD = 3
CTX = 96
BATCH = 48
MAX_STEPS = 2500
LR = 5e-4
WARMUP = 150
EVAL_EVERY = 250
SEED = 1337


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        hd = D // H
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp_fc = nn.Linear(d, 4 * d)
        self.mlp_out = nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp_out(F.gelu(self.mlp_fc(self.ln2(x))))
        return x


class SML(nn.Module):
    def __init__(self, vocab_size, d=D_MODEL, layers=N_LAYER, heads=N_HEAD, ctx=CTX):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=True)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


@torch.no_grad()
def sample(model, stoi, itos, prompt, max_new=110, temp=0.85, top_k=40, device="cpu"):
    model.eval()
    ids = [stoi.get(c, stoi[" "]) for c in prompt]
    ids = ids[-CTX:]
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / temp
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, 1).item()
        ch = itos[nid]
        if ch == "\n":
            break
        out.append(ch)
        ids.append(nid)
    model.train()
    return "".join(out)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = "cpu"
    torch.set_num_threads(2)

    with open(os.path.join(HERE, "corpus.txt"), encoding="utf-8") as f:
        text = f.read().lower()
    stoi, itos = build_vocab(text)
    V = len(stoi)
    print(f"vocab size: {V} chars: {''.join(sorted(stoi))!r}")
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.98 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"chars: train {len(train_data)}, val {len(val_data)}")

    model = SML(V).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(0, len(d) - CTX - 1, (BATCH,))
        x = torch.stack([d[i:i + CTX] for i in ix])
        y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
        return x.to(device), y.to(device)

    def lr_at(step):
        if step < WARMUP:
            return LR * step / WARMUP
        prog = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return LR * 0.5 * (1 + math.cos(math.pi * prog))

    log_path = os.path.join(HERE, "train_log.jsonl")
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step}/{MAX_STEPS} loss {loss.item():.4f} "
                  f"lr {pg['lr']:.2e} ({dt:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            with open(log_path, "a") as lf:
                lf.write(json.dumps({"step": step,
                                     "train_loss": round(loss.item(), 4)}) + "\n")
            for prompt in ["user: hi\nbot:", "user: bro\nbot:",
                           "user: how are you\nbot:"]:
                s = sample(model, stoi, itos, prompt)
                print(f"  sample [{prompt!r}] -> {s!r}", flush=True)
            ckpt = {"model": model.state_dict(),
                    "config": {"d_model": D_MODEL, "n_layer": N_LAYER,
                               "n_head": N_HEAD, "ctx": CTX,
                               "vocab": "".join(itos[i] for i in range(V)),
                               "n_params": n_params}}
            torch.save(ckpt, os.path.join(OUT, "final.pt"))
            print("  saved final.pt", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
