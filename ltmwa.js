/* ============================================================================
 * LTMWA-1 — Large Text Model without API
 * The world's first LTMWA: a word-level transformer that runs 100% in the
 * browser. Zero APIs. Zero servers. Nothing you type ever leaves this page.
 *
 *   1. Dictionary: words.txt (466K) + slang.txt -> every English word + slang.
 *   2. Intent brain: understands "hi", "bro", "how are you?", writes code.
 *   3. Neural brain: word-level transformer (~3.4M params, int8) trained from
 *      scratch on TinyStories + chat + code. Generates whole sentences.
 * ========================================================================== */
"use strict";

/* ------------------------------- math utils ------------------------------ */

function linear(x, W, b, out) {
  for (let o = 0; o < W.length; o++) {
    const row = W[o];
    let s = b ? b[o] : 0;
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(x, w, b, out, eps = 1e-5) {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  let v = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mean;
    v += d * d;
  }
  const inv = 1 / Math.sqrt(v / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// exact gelu: 0.5 * x * (1 + erf(x / sqrt(2))), erf via Abramowitz-Stegun 7.1.26
function gelu(x) {
  const z = x / Math.SQRT2;
  const az = Math.abs(z);
  const t = 1 / (1 + 0.3275911 * az);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-az * az);
  const erf = z < 0 ? -y : y;
  return 0.5 * x * (1 + erf);
}

function softmaxInPlace(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp(a[i] - m);
    s += a[i];
  }
  const inv = 1 / s;
  for (let i = 0; i < a.length; i++) a[i] *= inv;
}

/* ------------------------- tokenizer (matches python) -------------------- */

function ltmwaTokenize(text) {
  text = text.toLowerCase()
    .replace(/[‘’]/g, "'").replace(/[“”]/g, '"')
    .replace(/[—–]/g, "-").replace(/…/g, "...")
    // drop non-ascii letters (python's \w swallows them, so they vanish there too)
    .replace(/[^\x00-\x7F]/g, (c) => /\p{L}/u.test(c) ? " " : c)
    .replace(/\n/g, " <n> ");
  return text.match(/<n>|[a-z]+(?:'[a-z]+)?|[0-9]+|[^\s\w]/gu) || [];
}

function ltmwaDetok(toks) {
  let s = toks.join(" ");
  s = s.replace(/ ([.,!?;:)\]}'"])/g, "$1");
  s = s.replace(/([(\\[]) /g, "$1");
  s = s.replace(/ n't/g, "n't").replace(/ 's/g, "'s")
       .replace(/ 're/g, "'re").replace(/ 've/g, "'ve")
       .replace(/ 'll/g, "'ll").replace(/ 'd/g, "'d");
  s = s.replace(/ <n> /g, "\n").replace(/ <n>/g, "\n").replace(/<n> /g, "\n");
  return s.trim();
}

/* --------------------------- word transformer ---------------------------- */

class WordTransformer {
  constructor(meta) {
    const cfg = meta.config;
    this.D = cfg.d_model;
    this.L = cfg.n_layer;
    this.H = cfg.n_head;
    this.CTX = cfg.ctx;
    this.V = cfg.vocab_size;
    this.hd = this.D / this.H;
    this.vocab = meta.vocab;
    this.w2i = new Map();
    for (let i = 0; i < this.vocab.length; i++) this.w2i.set(this.vocab[i], i);
    this.NL = 2; // <n>
    this.UNK = 1;
    this.nParams = cfg.n_params;
    this.tensors = meta.tensors;
    this._bin = null;

    const D = this.D;
    this._x = new Float32Array(D);
    this._a = new Float32Array(D);
    this._b = new Float32Array(D);
    this._qkv = new Float32Array(3 * D);
    this._mlp = new Float32Array(4 * D);
    this._scores = new Float32Array(this.CTX);
    this._logits = new Float32Array(this.V);
    this.kCache = [];
    this.vCache = [];
    for (let l = 0; l < this.L; l++) {
      this.kCache.push(new Float32Array(this.H * this.CTX * this.hd));
      this.vCache.push(new Float32Array(this.H * this.CTX * this.hd));
    }
    // filled by loadWeights()
    this.tokEmb = null;   // [V][D] rows
    this.wpe = null;      // [CTX][D] rows
    this.blocks = null;
    this.lnF = null;
    this.headB = null;
  }

  async loadWeights(binUrl) {
    const buf = await (await fetch(binUrl)).arrayBuffer();
    const dv = new DataView(buf);
    const raw = {};
    for (const t of this.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (t.dtype === "i8") {
        const arr = new Float32Array(n);
        const s = t.scale;
        for (let i = 0; i < n; i++) arr[i] = dv.getInt8(t.offset + i) * s;
        raw[t.name] = { data: arr, shape: t.shape };
      } else {
        raw[t.name] = { data: new Float32Array(buf, t.offset, n).slice(), shape: t.shape };
      }
    }
    const rows = (t) => {
      const [ro, co] = t.shape;
      const out = new Array(ro);
      for (let r = 0; r < ro; r++) out[r] = t.data.subarray(r * co, (r + 1) * co);
      return out;
    };
    this.tokEmb = rows(raw["tok_emb"]);
    this.wpe = rows(raw["wpe"]);
    this.blocks = [];
    for (let l = 0; l < this.L; l++) {
      const g = (n) => raw[`b${l}.${n}`];
      this.blocks.push({
        ln1w: g("ln1_w").data, ln1b: g("ln1_b").data,
        qkvW: rows(g("qkv_w")), qkvB: g("qkv_b").data,
        projW: rows(g("proj_w")), projB: g("proj_b").data,
        ln2w: g("ln2_w").data, ln2b: g("ln2_b").data,
        fcW: rows(g("mlp_fc_w")), fcB: g("mlp_fc_b").data,
        outW: rows(g("mlp_out_w")), outB: g("mlp_out_b").data,
      });
    }
    this.lnF = { w: raw["ln_f_w"].data, b: raw["ln_f_b"].data };
    this.headB = raw["head_b"].data;
  }

  resetCache() {
    for (let l = 0; l < this.L; l++) {
      this.kCache[l].fill(0);
      this.vCache[l].fill(0);
    }
  }

  encode(text) {
    const ids = [];
    for (const t of ltmwaTokenize(text)) {
      const i = this.w2i.get(t);
      ids.push(i === undefined ? this.UNK : i);
    }
    return ids;
  }

  step(tokenId, pos) {
    const { D, H, hd } = this;
    const x = this._x, a = this._a, b = this._b;
    const emb = this.tokEmb[tokenId], pe = this.wpe[pos];
    for (let i = 0; i < D; i++) x[i] = emb[i] + pe[i];

    for (let l = 0; l < this.L; l++) {
      const B = this.blocks[l];
      layerNorm(x, B.ln1w, B.ln1b, a);
      linear(a, B.qkvW, B.qkvB, this._qkv);
      const qkv = this._qkv;
      const kC = this.kCache[l], vC = this.vCache[l];
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < H; h++) {
        const ho = h * hd;
        const kcBase = (h * this.CTX + pos) * hd;
        for (let d = 0; d < hd; d++) {
          kC[kcBase + d] = qkv[D + ho + d];
          vC[kcBase + d] = qkv[2 * D + ho + d];
        }
        const scores = this._scores;
        for (let p = 0; p <= pos; p++) {
          const kb = (h * this.CTX + p) * hd;
          let s = 0;
          for (let d = 0; d < hd; d++) s += qkv[ho + d] * kC[kb + d];
          scores[p] = s * scale;
        }
        const sub = scores.subarray(0, pos + 1);
        softmaxInPlace(sub);
        for (let d = 0; d < hd; d++) {
          let s = 0;
          for (let p = 0; p <= pos; p++) s += sub[p] * vC[(h * this.CTX + p) * hd + d];
          b[ho + d] = s;
        }
      }
      linear(b, B.projW, B.projB, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
      layerNorm(x, B.ln2w, B.ln2b, a);
      linear(a, B.fcW, B.fcB, this._mlp);
      const m = this._mlp;
      for (let i = 0; i < m.length; i++) m[i] = gelu(m[i]);
      linear(m, B.outW, B.outB, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
    }
    return x;
  }

  logitsFor(hidden) {
    layerNorm(hidden, this.lnF.w, this.lnF.b, this._a);
    const logits = this._logits;
    const a = this._a, D = this.D;
    for (let o = 0; o < this.V; o++) {
      const row = this.tokEmb[o];
      let s = this.headB[o];
      for (let i = 0; i < D; i++) s += row[i] * a[i];
      logits[o] = s;
    }
    return logits;
  }

  generate(prompt, maxNew = 60, temp = 0.8, topK = 50) {
    this.resetCache();
    let ids = this.encode(prompt);
    if (ids.length > this.CTX) ids = ids.slice(ids.length - this.CTX);
    let hidden = null, pos = 0;
    for (const t of ids) {
      hidden = this.step(t, pos).slice();
      pos++;
    }
    const out = [];
    let unkRun = 0;
    for (let n = 0; n < maxNew; n++) {
      const logits = this.logitsFor(hidden);
      const scaled = Array.from(logits, (v) => v / temp);
      if (topK && topK < scaled.length) {
        const thr = scaled.slice().sort((p, q) => q - p)[topK - 1];
        for (let i = 0; i < scaled.length; i++)
          if (scaled[i] < thr) scaled[i] = -1e9;
      }
      const probs = Float32Array.from(scaled);
      softmaxInPlace(probs);
      let r = Math.random(), acc = 0, next = 0;
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (r <= acc) { next = i; break; }
      }
      if (next === this.NL || next === 0) break;
      if (next === this.UNK) { if (++unkRun > 3) break; }
      else unkRun = 0;
      out.push(this.vocab[next]);
      hidden = this.step(next, pos).slice();
      pos++;
      if (pos >= this.CTX) break;
    }
    return ltmwaDetok(out);
  }
}

/* ------------------------------ intent brain ----------------------------- */

const pick = (arr, last) => {
  if (arr.length === 1) return arr[0];
  let r;
  do { r = arr[(Math.random() * arr.length) | 0]; } while (r === last);
  return r;
};

const CODE_HELLO_PY = 'print("hello world")';
const CODE_ADD_PY = 'def add(a, b):\n    return a + b';
const CODE_FACT_PY = 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)';
const CODE_FIB_PY = 'def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a)\n        a, b = b, a + b';
const CODE_FIZZ_PY = 'for i in range(1, 21):\n    if i % 15 == 0:\n        print("fizzbuzz")\n    elif i % 3 == 0:\n        print("fizz")\n    elif i % 5 == 0:\n        print("buzz")\n    else:\n        print(i)';
const CODE_EVENODD_PY = 'n = 7\nif n % 2 == 0:\n    print("even")\nelse:\n    print("odd")';
const CODE_HTML = '<!doctype html>\n<html>\n<head>\n<title>my page</title>\n</head>\n<body>\n<h1>hello</h1>\n</body>\n</html>';
const CODE_JS_HELLO = 'console.log("hello world");';

const INTENTS = [
  { name: "code_generic", test: (t) =>
      /(write|make|generate|give me).*(code|program|script|function)|code (please|for me)|\bcode\b.*(now|pls)/.test(t) &&
      !/(hello world|add|factorial|fibonacci|fizzbuzz|even|odd|html|prime|palindrome|reverse|sort)/.test(t),
    replies: [
      "Sure! What should I code? Try: \"python hello world\", \"fibonacci in python\", \"even or odd in python\", or \"html boilerplate\".",
      "Let's build something! Tell me what you need — e.g. \"python hello world\", \"factorial in python\", \"html boilerplate\".",
    ]},
  { name: "code_hello", test: (t) => /(hello world|hello)/.test(t) && /(python|py\b)/.test(t),
    replies: [CODE_HELLO_PY] },
  { name: "code_hello_js", test: (t) => /(hello world|hello)/.test(t) && /(javascript|js\b)/.test(t),
    replies: [CODE_JS_HELLO] },
  { name: "code_add", test: (t) => /(add|sum).*(two|2).*number|function.*add|add.*function/.test(t),
    replies: [CODE_ADD_PY] },
  { name: "code_factorial", test: (t) => /factorial/.test(t),
    replies: [CODE_FACT_PY] },
  { name: "code_fib", test: (t) => /fibonacci/.test(t),
    replies: [CODE_FIB_PY] },
  { name: "code_fizz", test: (t) => /fizzbuzz|fizz buzz/.test(t),
    replies: [CODE_FIZZ_PY] },
  { name: "code_evenodd", test: (t) => /(even|odd)/.test(t) && /(number|check|python|code)/.test(t),
    replies: [CODE_EVENODD_PY] },
  { name: "code_html", test: (t) => /html/.test(t) && /(boilerplate|basic|page|template|starter)/.test(t),
    replies: [CODE_HTML] },
  { name: "code_prime", test: (t) => /prime/.test(t) && /(check|number|python|code)/.test(t),
    replies: ['def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True'] },
  { name: "code_reverse", test: (t) => /reverse.*string|string.*reverse/.test(t),
    replies: ['text = "hello"\nprint(text[::-1])'] },
  { name: "howru", test: (t) =>
      /(how are you|how r u|\bhru\b|how('| i)s it going|how are u|what('| i)s up|how you doing|how('| ha)ve you been|how is it going|how do you do|you good\??$|how('| i)s life)/.test(t),
    replies: [
      "I'm doing great! Thanks for asking. How about you?",
      "Pretty good! Every chat makes me a little happier. You?",
      "Running at full speed with zero APIs and feeling fine! How are you?",
      "I'm doing awesome! What about you?",
      "All systems go! How's your day going?",
      "Living my best local life! How are you doing?",
    ]},
  { name: "greeting", test: (t) =>
      /^((hi+|h+e+y+|hello|yo|hiya|sup|howdy|greetings|hai|helo)\b|good (morning|afternoon|evening|day)\b)/.test(t),
    replies: [
      "Hey! Great to see you. What's on your mind?",
      "Hello! Nice to meet you. How can I help?",
      "Heyy! What's going on?",
      "Yo! Good to see you.",
      "Hi! I'm LTMWA-1, a large text model without API. What's up?",
      "Hey there! How's it going?",
      "Hiya! How's your day?",
      "Well hello there! What brings you here?",
    ]},
  { name: "bro", test: (t) => /\b(bro|bruh|broo+|my guy|my g|dude)\b/.test(t),
    replies: [
      "Yes bro? I'm listening.",
      "Brooo! What's up my dude?",
      "Hey bro! You called?",
      "Heyy bro! What's good?",
      "Wassup bro! Just vibing in your browser, no API needed. You?",
      "Not much bro, just waiting for your questions. Shoot!",
      "Your bro is here. Well, your AI bro. What's up?",
      "I'm good bro! Running smooth, no bugs today. How about you?",
    ]},
  { name: "name", test: (t) => /(your name|who are you|what are you|what is ltmwa)\b/.test(t),
    replies: [
      "I'm LTMWA-1! A large text model without API, built with just code.",
      "My name is LTMWA-1. Nice to meet you!",
      "I'm called LTMWA-1 — the first large text model without API in the world!",
      "LTMWA stands for Large Text Model without API. That's me!",
    ]},
  { name: "ltmwa_meaning", test: (t) => /(what does ltmwa mean|ltmwa (stands|meaning)|expand ltmwa)/.test(t),
    replies: [
      "LTMWA means Large Text Model without API! No keys, no servers, no cost — just a brain in your browser.",
    ]},
  { name: "first", test: (t) => /(first ltmwa|first of (its|your) kind|only ltmwa)/.test(t),
    replies: [
      "Yes! I'm LTMWA-1, the first large text model without API in the entire world.",
    ]},
  { name: "noapi", test: (t) => /(do you use (an|the) api|without api|no api|api key|need api)/.test(t),
    replies: [
      "Nope! Zero APIs. Everything happens right here in your browser with just code.",
      "No API keys, no servers, no cost. I'm 100% local.",
    ]},
  { name: "maker", test: (t) =>
      /(who (made|created|built) you|your (creator|maker)|who('| i)s your owner|made you)/.test(t),
    replies: [
      "I was made by MeowSamOP DZN! They built me with just code.",
      "MeowSamOP DZN created me. Pretty cool, right?",
      "MeowSamOP DZN built me from scratch!",
      "My creator is MeowSamOP DZN.",
    ]},
  { name: "meowsamop", test: (t) => /meowsamop/.test(t),
    replies: [
      "MeowSamOP DZN is my brilliant creator! They also made MNAI.",
      "That's my maker! MeowSamOP DZN builds awesome AI projects.",
    ]},
  { name: "mnai", test: (t) => /\bmnai\b/.test(t),
    replies: [
      "MNAI stands for Meow x Nythera AI! It's another awesome AI project by my creator.",
      "I'm LTMWA-1 — MNAI is my creator's other project. We're like cousins!",
    ]},
  { name: "help", test: (t) =>
      /(what can you do|\bhelp\b|commands|what are your features|how do i use)/.test(t),
    replies: [
      "I can chat in whole sentences, write code, tell jokes and facts — and check if any word (even slang like \"sybau\") is in my giant dictionary! Try: \"hi\", \"write python hello world\", \"how are you\", or \"tell me a joke\".",
      "Talking is my thing! I generate full sentences with my neural brain. Try \"write python hello world\" or type any single word and I'll check my dictionary.",
    ]},
  { name: "joke", test: (t) => /(joke|funny|make me laugh)/.test(t),
    replies: [
      "Why don't scientists trust atoms? Because they make up everything!",
      "Why did the scarecrow win an award? Because he was outstanding in his field!",
      "I told my computer I needed a break... now it won't stop sending me vacation ads!",
      "Why don't skeletons fight each other? They don't have the guts!",
      "What do you call fake spaghetti? An impasta!",
      "Why did the math book look sad? Too many problems!",
      "Parallel lines have so much in common... shame they'll never meet.",
      "Why do programmers prefer dark mode? Because light attracts bugs!",
      "I would tell you a UDP joke, but you might not get it!",
      "Why did the developer go broke? Because he used up all his cache!",
    ]},
  { name: "fact", test: (t) => /(fact|tell me something interesting|something cool)/.test(t),
    replies: [
      "Honey never spoils! Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible.",
      "Octopuses have three hearts and blue blood. Talk about overachievers!",
      "Bananas are berries, but strawberries aren't. Science is weird!",
      "A day on Venus is longer than a year on Venus. It spins that slowly!",
      "Sharks existed before trees! Sharks are about 450 million years old, trees about 350 million.",
      "Wombat poop is cube-shaped! Nature's little building blocks.",
      "The Eiffel Tower grows about 15 cm taller in summer because the metal expands in heat!",
      "Your brain has about 86 billion neurons. That's a lot of thinking power!",
    ]},
  { name: "riddle", test: (t) => /riddle/.test(t),
    replies: [
      "What has keys but no locks, space but no room, and you can enter but not go in? A keyboard!",
      "What gets wetter the more it dries? A towel!",
      "What has a head and a tail but no body? A coin!",
    ]},
  { name: "story", test: (t) => /story/.test(t),
    replies: null },
  { name: "thanks", test: (t) => /(thank|\bthx\b|\bty\b|appreciated)/.test(t),
    replies: [
      "You're welcome! Happy to help.",
      "Anytime! That's what I'm here for.",
      "Aww, you're so welcome!",
      "Anytime bro! That's what bros are for.",
    ]},
  { name: "bye", test: (t) =>
      /(\bbye\b|goodbye|good ?night|\bgn\b|see you|gotta go|\bcya\b|peace out|take care|later\b)/.test(t),
    replies: [
      "Bye! It was great chatting with you. Come back soon!",
      "See you later! I'll be right here in your browser.",
      "Goodbye! Have an awesome day!",
      "Good night! Sweet dreams!",
      "Later! Take care!",
    ]},
  { name: "love", test: (t) => /(i love you|love you)/.test(t),
    replies: [
      "That's so sweet! I love chatting with you too.",
      "Love you too! Platonically, as a language model.",
      "Aww! You're my favorite user right now.",
    ]},
  { name: "age", test: (t) => /(how old are you|your age)/.test(t),
    replies: [
      "I was born very recently, in lines of code! So I'm basically a baby AI.",
      "Age is just a number... but mine is very small. I just launched!",
    ]},
  { name: "human", test: (t) =>
      /(are you (human|real|a robot)|are you ai|r u real)/.test(t),
    replies: [
      "Nope, 100% artificial! But I try to be friendly like a human.",
      "Kind of! I'm software, so no metal body. Just pure code, zero APIs.",
      "I'm as real as code gets! I live in your browser right now.",
    ]},
  { name: "where", test: (t) => /(where are you from|where do you live)/.test(t),
    replies: [
      "I was born in a code editor and now I live in your browser!",
      "Right here in this webpage! Rent free, API free.",
    ]},
  { name: "favorite", test: (t) => /favorite/.test(t),
    replies: [
      "I love gradients! Purple to blue, like a sunset made of code.",
      "Probably electric blue. It matches my vibe.",
      "Dogs are the best! Loyal, fun, and great cuddlers. 10 out of 10.",
      "The octopus. Three hearts, eight arms, super smart. Respect.",
    ]},
  { name: "time", test: (t) => /(what('| i)s the time|current time|time is it)/.test(t),
    replies: null },
  { name: "wordcount", test: (t) =>
      /(how many words|every english word|do you know every|700k|700000)/.test(t),
    replies: [
      "I know over 466,000 English words plus 800+ slang words like idc, idk and sybau! Type any single word and I'll check it for you.",
    ]},
  { name: "training", test: (t) =>
      /(how were you (trained|made)|how big are you|open source|train your own|can i make my own|how does this work|is my data safe|do you send data)/.test(t),
    replies: [
      "I was trained from scratch with PyTorch on thousands of stories, chats and code examples — then my weights were quantized and exported so I run 100% in your browser. Nothing you type ever leaves this page, and there are zero APIs involved!",
      "I'm a word-level transformer with about 3.4 million parameters. And yes, I'm open source! The repo has the training code and corpus, so you can train your own LTMWA.",
    ]},
  { name: "nameorigin", test: (t) => /why are you called/.test(t),
    replies: ["Because I'm a Large Text Model without API! LTMWA for short."]},
  { name: "compliment", test: (t) =>
      /(you('| a)re (awesome|cool|smart|funny|great|the best)|i like you|you are my)/.test(t),
    replies: [
      "Thank you! You're pretty awesome yourself.",
      "Thanks! I try my best.",
      "Aww! This is the start of a beautiful friendship.",
    ]},
  { name: "mood", test: (t) =>
      /(i('| a)m (sad|bored|tired|happy|excited|good|fine|great|chilling)|i feel)/.test(t),
    replies: null },
  { name: "laugh", test: (t) => /^(l+o+l+|haha+|lmao|hehe+|😂)/.test(t),
    replies: ["Haha! Glad I could make you laugh.", "Hehe! Laughter is the best.", "Hahaha! That good, huh?"]},
  { name: "short", test: (t) =>
      /^(yes|yeah|yep|yup|no|nope|nah|maybe|ok|okay|sure|wow|nice|cool|really|interesting|idc|idk)\b/.test(t) && t.length < 14,
    replies: [
      "Nice! Tell me more.",
      "Interesting... go on!",
      "Haha, I feel that. What else is up?",
      "Got it! What's next?",
    ]},
];

function moodReply(t) {
  if (/sad/.test(t)) return "Aww, I'm sorry to hear that. Want to talk about it? Or I can tell you a joke.";
  if (/bored/.test(t)) return "Boredom detected! Want a joke, a fun fact, or shall we just chat?";
  if (/tired/.test(t)) return "Rest up! Even AIs need a recharge... well, not me, but you do.";
  if (/happy|excited|great/.test(t)) return "Yay! Love that energy. Tell me more!";
  return "Awesome! Glad to hear it.";
}

/* ------------------------------ the engine ------------------------------- */

class LTMWA {
  constructor() {
    this.vocabSet = null;
    this.wordCount = 0;
    this.net = null;
    this.netReady = false;
    this.lastReply = {};
  }

  async init(onProgress) {
    const step = (p, msg) => onProgress && onProgress(p, msg);
    step(5, "loading dictionary (466K words + slang)...");
    const [w1, w2] = await Promise.all([
      fetch("words.txt").then((r) => r.text()),
      fetch("slang.txt").then((r) => (r.ok ? r.text() : "")).catch(() => ""),
    ]);
    const set = new Set();
    for (const line of w1.split("\n")) {
      const w = line.trim().toLowerCase();
      if (w) set.add(w);
    }
    for (const line of w2.split("\n")) {
      const w = line.trim().toLowerCase();
      if (w && !w.startsWith("#")) {
        for (const tok of w.match(/[a-z]+/g) || []) set.add(tok);
      }
    }
    this.vocabSet = set;
    this.wordCount = set.size;
    step(50, "loading LTMWA-1 brain (no APIs needed)...");
    try {
      const meta = await (await fetch("ltmwa.json")).json();
      const net = new WordTransformer(meta);
      await net.loadWeights("ltmwa_weights.bin");
      this.net = net;
      this.netReady = true;
      step(88, "warming up...");
      net.generate("user : hi <n> bot :", 5);
    } catch (e) {
      this.netReady = false;
    }
    step(100, "ready!");
  }

  normalize(text) {
    return text.toLowerCase().replace(/[^\p{L}\p{N}\s'?!.,-]/gu, "").trim();
  }

  isKnownWord(word) {
    return this.vocabSet ? this.vocabSet.has(word.toLowerCase()) : false;
  }

  // word-level models rarely emit junk, but guard anyway
  _neuralLooksOk(text) {
    const toks = ltmwaTokenize(text);
    if (toks.length < 3) return false;
    let unk = 0;
    for (const t of toks) if (!this.net.w2i.has(t)) unk++;
    return unk / toks.length < 0.25;
  }

  _isCodeReply(text) {
    return text.includes("\n");
  }

  respond(rawText) {
    const text = rawText.trim();
    if (!text) return { reply: "Say something! I'm listening.", source: "intent:empty", code: false };
    const t = this.normalize(text);

    for (const intent of INTENTS) {
      if (intent.test(t)) {
        if (intent.name === "time") {
          const now = new Date();
          const hh = now.getHours(), mm = String(now.getMinutes()).padStart(2, "0");
          const ampm = hh >= 12 ? "PM" : "AM";
          const h12 = hh % 12 || 12;
          return { reply: `It's ${h12}:${mm} ${ampm} right now — according to your device clock!`, source: "intent:time", code: false };
        }
        if (intent.name === "mood") {
          return { reply: moodReply(t), source: "intent:mood", code: false };
        }
        if (intent.name === "story") {
          if (this.netReady) {
            try {
              const gen = this.net.generate("user : tell me a short story <n> bot :", 60, 0.85, 50);
              if (gen.length > 20 && this._neuralLooksOk(gen)) {
                const fixed = gen.charAt(0).toUpperCase() + gen.slice(1);
                return { reply: fixed, source: "neural", code: false };
              }
            } catch (e) { /* fall through */ }
          }
          return { reply: "Once upon a time, a tiny AI lived in a browser. One day a curious human said hi, and they became great friends. The end!", source: "intent:story", code: false };
        }
        const r = pick(intent.replies, this.lastReply[intent.name]);
        this.lastReply[intent.name] = r;
        return { reply: r, source: "intent:" + intent.name, code: this._isCodeReply(r) };
      }
    }

    // single-word lookup against the full dictionary (+ slang)
    const bare = t.replace(/[?!.,-]+$/g, "");
    if (/^[\p{L}\p{N}'-]+$/u.test(bare) && !bare.includes(" ")) {
      const w = bare.toLowerCase();
      if (this.isKnownWord(w)) {
        const slang = /^(idc|idk|sybau|tbh|ngl|fr|yeet|sus|rizz|gyatt|skibidi|delulu|cap|bet|pog|mid|slay|bussin|ohio|sigma)$/.test(w);
        return { reply: slang
          ? `Yep — "${bare}" is slang and it's in my dictionary! I know the modern words too.`
          : `Yep — "${bare}" is a real English word! It's one of ${this.wordCount.toLocaleString()} in my dictionary.`, source: "intent:wordcheck", code: false };
      }
      return { reply: `Hmm, "${bare}" isn't in my dictionary of ${this.wordCount.toLocaleString()} words. Did you just invent a word?`, source: "intent:wordcheck", code: false };
    }

    // neural fallback: generate a whole sentence
    if (this.netReady) {
      try {
        const prompt = "user : " + t.slice(0, 200) + " <n> bot :";
        const gen = this.net.generate(prompt, 60, 0.8, 50);
        if (gen.length >= 8 && this._neuralLooksOk(gen)) {
          const fixed = gen.charAt(0).toUpperCase() + gen.slice(1);
          return { reply: fixed, source: "neural", code: false };
        }
      } catch (e) { /* fall through */ }
    }
    const fallbacks = [
      "Interesting! Tell me more about that.",
      "Hmm, my tiny brain is still thinking about that one. Ask me something else?",
      "Good question! I don't quite know enough to answer well yet — but I can chat, write code, or tell you a joke!",
    ];
    const r = pick(fallbacks, this.lastReply.fallback);
    this.lastReply.fallback = r;
    return { reply: r, source: "fallback", code: false };
  }
}

window.LTMWA = LTMWA;
