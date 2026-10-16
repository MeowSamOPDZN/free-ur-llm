"""Train LTMWA-1: word-level transformer from scratch on CPU.

V=8192, D=192, L=4, H=4, CTX=128, tied input/output embeddings.
Corpus: TinyStories + chat + code + identity + slang (ltmwa_ids.pt).
Exports checkpoints/ltmwa_final.pt for export_ltmwa.py.
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

D_MODEL, N_LAYER, N_HEAD, CTX = 192, 4, 4, 128
BATCH, MAX_STEPS, LR, WARMUP = 16, 3000, 4e-4, 200
EVAL_EVERY, SEED = 300, 7


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.heads, D // self.heads
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = F.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
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


class LTMWA(nn.Module):
    def __init__(self, vocab_size, d=D_MODEL, layers=N_LAYER, heads=N_HEAD, ctx=CTX):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_b = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return x @ self.tok_emb.weight.t() + self.head_b  # tied head


@torch.no_grad()
def sample(model, stoi, itos, prompt, max_new=40, temp=0.8, top_k=50):
    model.eval()
    ids = [stoi.get(t, stoi["<unk>"]) for t in prompt.split(" ")]
    ids = ids[-CTX:]
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-CTX:]], dtype=torch.long)
        logits = model(x)[0, -1] / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        nid = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if nid == stoi["<n>"]:
            break
        out.append(itos[nid])
        ids.append(nid)
    model.train()
    return detok(out)


def detok(toks):
    s = " ".join(toks)
    for p in (".", ",", "!", "?", ";", ":", ")", "]", "}", "'", '"'):
        s = s.replace(" " + p, p)
    s = s.replace("( ", "(").replace("[ ", "[")
    s = s.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re")
    return s.strip()


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    with open(os.path.join(HERE, "ltmwa_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)["tokens"]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = vocab
    V = len(vocab)
    print(f"vocab: {V}", flush=True)

    data = torch.load(os.path.join(HERE, "ltmwa_ids.pt")).long()
    val = torch.load(os.path.join(HERE, "ltmwa_val_ids.pt")).long()
    print(f"tokens: train {len(data):,}, val {len(val):,}", flush=True)

    model = LTMWA(V)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def get_batch(d):
        ix = torch.randint(0, len(d) - CTX - 1, (BATCH,))
        x = torch.stack([d[i:i + CTX] for i in ix])
        y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        for _ in range(20):
            xb, yb = get_batch(val)
            tot += F.cross_entropy(model(xb).view(-1, V), yb.view(-1)).item()
            n += 1
        model.train()
        return tot / n

    def lr_at(step):
        if step < WARMUP:
            return LR * step / WARMUP
        prog = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return LR * 0.5 * (1 + math.cos(math.pi * prog))

    log_path = os.path.join(HERE, "ltmwa_log.jsonl")
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        xb, yb = get_batch(data)
        loss = F.cross_entropy(model(xb).view(-1, V), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step}/{MAX_STEPS} loss {loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = eval_loss()
            with open(log_path, "a") as lf:
                lf.write(json.dumps({"step": step,
                                     "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl, 4)}) + "\n")
            print(f"  eval: train {loss.item():.4f} val {vl:.4f}", flush=True)
            for p in ["user : hi <n> bot :",
                      "user : write python hello world <n> bot :",
                      "user : who made you <n> bot :"]:
                print(f"  sample [{p}] -> {sample(model, stoi, itos, p)!r}",
                      flush=True)
            torch.save({"model": model.state_dict(),
                        "config": {"d_model": D_MODEL, "n_layer": N_LAYER,
                                   "n_head": N_HEAD, "ctx": CTX,
                                   "vocab_size": V,
                                   "n_params": n_params}},
                       os.path.join(OUT, "ltmwa_final.pt"))
            print("  saved ltmwa_final.pt", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
