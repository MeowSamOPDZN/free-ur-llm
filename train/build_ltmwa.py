"""Build LTMWA word-level corpus + vocabulary.

Outputs:
    ltmwa_vocab.json  - {"tokens": [8192 strings], "pad":0, "unk":1, "nl":2}
    ltmwa_ids.pt      - torch int32 tensor of token ids (train)
    ltmwa_val_ids.pt  - torch int32 tensor of token ids (val)

Tokenizer (MUST match ltmwa.js exactly):
    lowercase -> normalize quotes/dashes -> "\\n" becomes " <n> "
    -> regex: <n> | [a-z]+('[a-z]+)? | [0-9]+ | any other single char
"""
import json
import os
import random
import re
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NANO = os.path.expanduser("~/workspace/mnai-nano")
VOCAB_SIZE = 8192
PAD, UNK, NL = 0, 1, 2

TOKEN_RE = re.compile(r"<n>|[a-z]+(?:'[a-z]+)?|[0-9]+|[^\s\w]")


def normalize(text):
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u2026", "...")
    return text


def tokenize(text):
    text = normalize(text).replace("\n", " <n> ")
    return TOKEN_RE.findall(text)


def load_blocks(path):
    """Parse user:/bot: blocks (continuation lines allowed)."""
    pairs = []
    user_lines, bot_lines, cur = [], [], None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("user:"):
                if user_lines or bot_lines:
                    pairs.append((" ".join(user_lines), " ".join(bot_lines)))
                user_lines, bot_lines = [line[5:].strip()], []
                cur = "u"
            elif line.startswith("bot:"):
                bot_lines = [line[4:].strip()]
                cur = "b"
            elif line.strip() == "":
                if user_lines or bot_lines:
                    pairs.append((" ".join(user_lines), " ".join(bot_lines)))
                    user_lines, bot_lines = [], []
                cur = None
            elif cur == "u":
                user_lines.append(line.strip())
            elif cur == "b":
                bot_lines.append(line.strip())
    if user_lines or bot_lines:
        pairs.append((" ".join(user_lines), " ".join(bot_lines)))
    return [(u, b) for u, b in pairs if u and b]


IDENTITY = [
    ("what is your name", "i am ltmwa-1, a large text model without api."),
    ("whats your name", "ltmwa-1! large text model without api."),
    ("your name", "i am called ltmwa-1. you can call me ltmwa!"),
    ("who are you", "i am ltmwa-1, a large text model without api. i run with zero apis, just code!"),
    ("tell me about yourself", "i am ltmwa-1! a large text model without api. i can chat, write code, and i know every english word in my dictionary."),
    ("who made you", "i was made by meowsamop dzn!"),
    ("who created you", "meowsamop dzn created me!"),
    ("who built you", "meowsamop dzn built me from scratch!"),
    ("your creator", "my creator is meowsamop dzn."),
    ("your maker", "meowsamop dzn made me!"),
    ("what does ltmwa mean", "ltmwa means large text model without api!"),
    ("what is ltmwa", "ltmwa stands for large text model without api. it is a new kind of ai that runs with zero apis!"),
    ("are you the first ltmwa", "yes! i am ltmwa-1, the first large text model without api in the world."),
    ("is this the first ltmwa", "yes! ltmwa-1 is the first of its kind. no apis, just a tiny brain in your browser."),
    ("do you use an api", "nope! zero apis. everything happens right here in your browser."),
    ("are you free", "100 percent free! no api keys, no servers, no cost."),
    ("why no api", "because apis need keys and servers! i do everything locally with just code."),
]

SLANG_CHAT = [
    ("idk what to do today", "tbh same! want a joke, a fun fact, or shall we just chat?"),
    ("idc tbh", "fair enough! no cap, sometimes not caring is self care."),
    ("sybau", "sybau?? okay okay, i will! what's up?"),
    ("ngl you are cool", "ngl that means a lot! you are cool too fr."),
    ("fr?", "fr fr!"),
    ("that's cap", "no cap here! i only speak facts."),
    ("say less", "less. okay done."),
    ("bet", "bet! what are we doing?"),
    ("it's giving", "it's giving... tiny ai with big dreams!"),
    ("slay", "you slay! period."),
    ("understood the assignment", "i understood the assignment! what's next?"),
    ("touch grass", "i would if i had legs! you go touch grass for both of us."),
    ("doomscrolling again", "put the phone down! or keep scrolling, i'm not your mom."),
    ("brainrot", "certified brainrot survivor right here."),
]


def main():
    random.seed(7)
    texts = []

    # 1. TinyStories — teaches fluent whole sentences
    print("loading TinyStories...", flush=True)
    with open(os.path.join(NANO, "train.txt"), encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("<|endoftext|>", "\n")
    texts.append(("story", raw, 1))

    # 2. chat corpus x3 — teaches conversation + user:/bot: format
    print("loading chat pairs...", flush=True)
    chat = []
    for fn in ("corpus.txt",):
        for u, b in load_blocks(os.path.join(HERE, fn)):
            chat.append(f"user: {u}\nbot: {b}\n")
    # rebrand SML mentions -> LTMWA inside chat text
    chat = [c.replace("freeur llm", "ltmwa-1").replace("freeur", "ltmwa") for c in chat]
    texts.append(("chat", "\n".join(chat), 3))

    # 3. code pairs x5 — teaches code generation
    print("loading code pairs...", flush=True)
    code = []
    for u, b in load_blocks(os.path.join(HERE, "code_pairs.txt")):
        code.append(f"user: {u}\nbot: {b}\n")
    texts.append(("code", "\n".join(code), 5))

    # 4. identity x30
    ident = []
    for _ in range(30):
        for u, b in IDENTITY:
            ident.append(f"user: {u}\nbot: {b}\n")
    texts.append(("identity", "\n".join(ident), 1))

    # 5. slang chat x8
    slangc = []
    for _ in range(8):
        for u, b in SLANG_CHAT:
            slangc.append(f"user: {u}\nbot: {b}\n")
    texts.append(("slang", "\n".join(slangc), 1))

    # tokenize + count
    print("tokenizing...", flush=True)
    counter = Counter()
    tokenized_parts = []
    for name, text, rep in texts:
        toks = tokenize(text)
        counter.update(toks)
        tokenized_parts.append((name, toks, rep))
        print(f"  {name}: {len(toks):,} tokens x{rep}", flush=True)

    # build vocab: most common, force-include slang + special words
    slang_words = set()
    with open(os.path.join(HERE, "slang.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            for w in re.findall(r"[a-z]+", line):
                slang_words.add(w)
    forced = set(slang_words) | {"ltmwa", "meowsamop", "dzn", "sybau", "idc", "idk"}
    forced = {w for w in forced if w not in ("<n>", "<pad>", "<unk>")}

    vocab = ["<pad>", "<unk>", "<n>"]
    seen = set(vocab) | forced
    # add forced words first (after specials)
    for w in sorted(forced):
        vocab.append(w)
    for w, _ in counter.most_common():
        if w in seen or w in ("<pad>", "<unk>"):
            continue
        if len(vocab) >= VOCAB_SIZE:
            break
        vocab.append(w)
        seen.add(w)
    assert len(vocab) == VOCAB_SIZE, len(vocab)
    print(f"vocab: {len(vocab)} (forced {len(forced)} incl. slang)", flush=True)
    assert "ltmwa" in vocab and "sybau" in vocab and "idc" in vocab

    stoi = {w: i for i, w in enumerate(vocab)}
    with open(os.path.join(HERE, "ltmwa_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"tokens": vocab, "pad": PAD, "unk": UNK, "nl": NL}, f)

    # encode corpus
    print("encoding...", flush=True)
    all_ids = []
    for name, toks, rep in tokenized_parts:
        ids = [stoi.get(t, UNK) for t in toks]
        all_ids.extend(ids * rep)
        print(f"  {name}: {len(ids) * rep:,} ids", flush=True)
    ids_t = torch.tensor(all_ids, dtype=torch.int32)
    print(f"total: {len(ids_t):,} tokens", flush=True)
    n_val = int(0.02 * len(ids_t))
    torch.save(ids_t[:-n_val] if n_val else ids_t,
               os.path.join(HERE, "ltmwa_ids.pt"))
    if n_val:
        torch.save(ids_t[-n_val:], os.path.join(HERE, "ltmwa_val_ids.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
