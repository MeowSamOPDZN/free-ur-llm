# LTMWA-1 — Large Text Model without API

**The world's first LTMWA.** A word-level transformer that runs **100% in your
browser**. Zero APIs. Zero servers. Zero API keys. Nothing you type ever leaves
the page.

🌐 **Live demo:** https://free-ur-llm.vercel.app/

## What it is

LTMWA-1 has two brains:

1. **Intent brain** (`ltmwa.js`) — understands greetings and reacts instantly:
   `Hi`, `Bro`, `How are you?` — plus jokes, facts, stories, riddles,
   dictionary lookups (including slang like `sybau`), and **instant code
   generation** (`write python hello world`, `fibonacci in python`,
   `html boilerplate`...).
2. **Neural brain** (`ltmwa.json` + `ltmwa_weights.bin`) — a **word-level
   transformer** (~3.4M params, int8 quantized) trained **from scratch** with
   PyTorch on TinyStories + 4,000 chat conversations + 150 code examples.
   It generates **whole sentences** for everything else.

Dictionary: `words.txt` (466K English words) + `slang.txt` (800+ modern slang
words the old dictionary missed: idc, idk, sybau, rizz, ...). Type any single
word and it tells you if it's real.

## Try it

- `Hi` / `Bro` / `How are you?`
- `Write python hello world`
- `Fibonacci in python`
- `Tell me a story` (neural — it writes one itself!)
- `Is "sybau" a word?`
- `Who made you?` / `What does LTMWA mean?`

## Project layout

```
free-ur-llm/
├── index.html            # LTMWA-1 chat UI (dark aesthetic)
├── ltmwa.js              # engine: tokenizer, intent brain, transformer inference
├── ltmwa.json            # model config + vocab + tensor map
├── ltmwa_weights.bin     # int8 quantized weights (~3.4 MB)
├── words.txt             # 466,546 English words
├── slang.txt             # 800+ slang words
└── train/
    ├── train_ltmwa.py    # train the word-level transformer from scratch (CPU)
    ├── build_ltmwa.py    # builds corpus + 8192-word vocab
    ├── export_ltmwa.py   # int8 export -> ltmwa.json + ltmwa_weights.bin
    ├── code_pairs.txt    # code Q&A training data
    ├── slang.txt         # slang dictionary
    └── seed_pairs*.txt   # chat training data
```

## Train your own LTMWA

```bash
cd train
python build_ltmwa.py   # TinyStories + chat + code -> corpus + vocab (8k)
python train_ltmwa.py   # train from scratch on CPU (~3-4 hrs, 3000 steps)
python export_ltmwa.py  # int8 quantize -> ../ltmwa.json + ../ltmwa_weights.bin
```

Open `index.html` — your freshly trained brain is live. Edit the data to
change its personality, add your own Q&A, retrain.

Model: word-level transformer, vocab 8192, `d_model=192`, 4 layers, 4 heads,
ctx 128, tied embeddings (~3.4M params). The JS inference in `ltmwa.js` mirrors
the PyTorch model exactly (verified numerically against torch).

## The honest version

This is a *real* tiny neural language model, trained from random
initialization — but it's honest about its size. A few million parameters
trained on a few million tokens is great at simple sentences, short stories
and small code snippets, and the intent brain handles greetings, code requests
and word lookups perfectly every time. No fake "AGI" claims: it's a genuinely
small model that genuinely runs with zero APIs.

## Credits

Made by **MeowSamOP DZN**. LTMWA = Large Text Model without API.
