/* ============================================================================
 * FreeUr LLM (SML-1) — engine
 * A tiny language model that runs 100% in the browser:
 *   1. words.txt  -> the full English dictionary (466,550 words), loaded as
 *      the vocabulary the model "knows".
 *   2. Intent brain -> understands greetings ("hi", "bro", "how are you?")
 *      and reacts with natural replies.
 *   3. Neural brain -> a tiny char-level transformer (~350K params) trained
 *      from scratch on chat conversations, for everything else.
 * No servers. Nothing you type ever leaves this page.
 * ========================================================================== */
"use strict";

/* ------------------------------- math utils ------------------------------ */

function linear(x, W, b, out) {
  // x: Float32Array[in], W: [out][in], b: [out] -> out: Float32Array[out]
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

// exact gelu via erf approximation (Abramowitz & Stegun 7.1.26)
function gelu(x) {
  const c = 0.7978845608028654; // sqrt(2/pi)
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  const erf = x < 0 ? -y : y;
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

/* --------------------------- tiny transformer ---------------------------- */

class TinyTransformer {
  constructor(bundle) {
    const cfg = bundle.config;
    this.D = cfg.d_model;
    this.L = cfg.n_layer;
    this.H = cfg.n_head;
    this.CTX = cfg.ctx;
    this.vocab = cfg.vocab; // string of chars, index = token id
    this.V = this.vocab.length;
    this.hd = this.D / this.H;
    this.W = bundle.weights;
    this.nParams = cfg.n_params;

    // scratch buffers
    const D = this.D;
    this._x = new Float32Array(D);
    this._a = new Float32Array(D);
    this._b = new Float32Array(D);
    this._qkv = new Float32Array(3 * D);
    this._mlp = new Float32Array(4 * D);
    this._scores = new Float32Array(this.CTX);
    this._logits = new Float32Array(this.V);
    // kv cache: per layer [H * CTX * hd]
    this.kCache = [];
    this.vCache = [];
    for (let l = 0; l < this.L; l++) {
      this.kCache.push(new Float32Array(this.H * this.CTX * this.hd));
      this.vCache.push(new Float32Array(this.H * this.CTX * this.hd));
    }
  }

  resetCache() {
    for (let l = 0; l < this.L; l++) {
      this.kCache[l].fill(0);
      this.vCache[l].fill(0);
    }
  }

  // run one token at position pos; returns hidden state (Float32Array D) — reused buffer!
  step(tokenId, pos) {
    const { D, H, hd, W } = this;
    const x = this._x, a = this._a, b = this._b;
    const emb = W.tok_emb[tokenId], pe = W.wpe[pos];
    for (let i = 0; i < D; i++) x[i] = emb[i] + pe[i];

    for (let l = 0; l < this.L; l++) {
      const B = W.blocks[l];
      // --- attention sub-block ---
      layerNorm(x, B.ln1_w, B.ln1_b, a);
      linear(a, B.qkv_w, B.qkv_b, this._qkv);
      const qkv = this._qkv;
      const kC = this.kCache[l], vC = this.vCache[l];
      const scale = 1 / Math.sqrt(hd);
      // per head
      for (let h = 0; h < H; h++) {
        const ho = h * hd;
        // write k,v of this token into cache
        const kcBase = (h * this.CTX + pos) * hd;
        for (let d = 0; d < hd; d++) {
          kC[kcBase + d] = qkv[D + ho + d];
          vC[kcBase + d] = qkv[2 * D + ho + d];
        }
        // scores over 0..pos
        const scores = this._scores;
        for (let p = 0; p <= pos; p++) {
          const kb = (h * this.CTX + p) * hd;
          let s = 0;
          for (let d = 0; d < hd; d++) s += qkv[ho + d] * kC[kb + d];
          scores[p] = s * scale;
        }
        const sub = scores.subarray(0, pos + 1);
        softmaxInPlace(sub);
        // weighted sum of v
        for (let d = 0; d < hd; d++) {
          let s = 0;
          for (let p = 0; p <= pos; p++) {
            s += sub[p] * vC[(h * this.CTX + p) * hd + d];
          }
          b[ho + d] = s; // reuse b as attention output
        }
      }
      linear(b, B.proj_w, B.proj_b, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
      // --- mlp sub-block ---
      layerNorm(x, B.ln2_w, B.ln2_b, a);
      linear(a, B.mlp_fc_w, B.mlp_fc_b, this._mlp);
      const m = this._mlp;
      for (let i = 0; i < m.length; i++) m[i] = gelu(m[i]);
      linear(m, B.mlp_out_w, B.mlp_out_b, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
    }
    return x;
  }

  logitsFor(hidden) {
    const W = this.W;
    layerNorm(hidden, W.ln_f_w, W.ln_f_b, this._a);
    return linear(this._a, W.head_w, W.head_b, this._logits);
  }

  encode(str) {
    const ids = [];
    for (const ch of str) {
      const i = this.vocab.indexOf(ch);
      ids.push(i >= 0 ? i : this.vocab.indexOf(" "));
    }
    return ids;
  }

  generate(prompt, maxNew = 110, temp = 0.9, topK = 40) {
    this.resetCache();
    let ids = this.encode(prompt);
    if (ids.length > this.CTX) ids = ids.slice(ids.length - this.CTX);
    let hidden = null, pos = 0;
    for (const t of ids) {
      hidden = this.step(t, pos).slice();
      pos++;
    }
    let out = "";
    for (let n = 0; n < maxNew; n++) {
      const logits = this.logitsFor(hidden);
      const scaled = Array.from(logits, (v) => v / temp);
      // top-k
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
      const ch = this.vocab[next];
      if (ch === "\n") break;
      out += ch;
      hidden = this.step(next, pos).slice();
      pos++;
      if (pos >= this.CTX) break;
    }
    return out.trim();
  }
}

/* ------------------------------ intent brain ----------------------------- */

const pick = (arr, last) => {
  if (arr.length === 1) return arr[0];
  let r;
  do { r = arr[(Math.random() * arr.length) | 0]; } while (r === last);
  return r;
};

const INTENTS = [
  { name: "howru", test: (t) =>
      /(how are you|how r u|\bhru\b|how('| i)s it going|how are u|what('| i)s up|how you doing|how('| ha)ve you been|how is it going|how do you do|you good\??$|how('| i)s life)/.test(t),
    replies: [
      "I'm doing great! Thanks for asking. How about you?",
      "Pretty good! Every chat makes me a little happier. You?",
      "Running at full speed and feeling fine! How are you?",
      "I'm doing awesome! What about you?",
      "All systems go! How's your day going?",
      "Living my best digital life! How are you doing?",
    ]},
  { name: "greeting", test: (t) =>
      /^((hi+|h+e+y+|hello|yo|hiya|sup|howdy|greetings|hai|helo)\b|good (morning|afternoon|evening|day)\b)/.test(t),
    replies: [
      "Hey! Great to see you. What's on your mind?",
      "Hello! Nice to meet you. How can I help?",
      "Heyy! What's going on?",
      "Yo! Good to see you.",
      "Hi! I'm FreeUr LLM, a tiny AI built with just code. What's up?",
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
      "Wassup bro! Just vibing in the code. You?",
      "Not much bro, just waiting for your questions. Shoot!",
      "Your bro is here. Well, your AI bro. What's up?",
      "I'm good bro! Running smooth, no bugs today. How about you?",
    ]},
  { name: "name", test: (t) => /(your name|who are you|what are you)\b/.test(t),
    replies: [
      "I'm FreeUr LLM! A small language model built with just code.",
      "My name is FreeUr LLM. Nice to meet you!",
      "I'm called FreeUr LLM — you can call me Free!",
      "Just a small language model with big dreams. I'm FreeUr LLM!",
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
      "Nope, I'm FreeUr LLM — MNAI is my creator's other project. We're like cousins!",
    ]},
  { name: "help", test: (t) =>
      /(what can you do|\bhelp\b|commands|what are your features|how do i use)/.test(t),
    replies: [
      "I can chat with you, answer questions, tell jokes and facts — and check if any word is real English from my giant dictionary! Try: \"hi\", \"bro\", \"how are you\", or \"tell me a joke\".",
      "Talking is my thing! Ask me anything, or try \"tell me a fun fact\". Type any single English word and I'll tell you if it's in my dictionary of 466,550 words.",
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
      "Debugged my code today... turns out the bug was me all along.",
      "I would tell you a UDP joke, but you might not get it!",
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
    replies: [
      "Once upon a time, a tiny AI lived in a browser. One day a curious human said hi, and they became great friends. The end!",
      "There once was a chatbot who dreamed of understanding every word. So it read a dictionary of 466,550 words and lived happily ever after!",
    ]},
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
      "Kind of! I'm software, so no metal body. Just pure code.",
      "I'm as real as code gets! I live in your browser right now.",
    ]},
  { name: "where", test: (t) => /(where are you from|where do you live)/.test(t),
    replies: [
      "I was born in a code editor and now I live in your browser!",
      "Right here in this webpage! Rent free.",
    ]},
  { name: "favorite", test: (t) => /favorite/.test(t),
    replies: [
      "I love gradients! Purple to blue, like a sunset made of code.",
      "Probably electric blue. It matches my vibe.",
      "Dogs are the best! Loyal, fun, and great cuddlers. 10 out of 10.",
      "The octopus. Three hearts, eight arms, super smart. Respect.",
    ]},
  { name: "time", test: (t) => /(what('| i)s the time|current time|time is it)/.test(t),
    replies: null }, // handled dynamically
  { name: "wordcount", test: (t) =>
      /(how many words|every english word|do you know every)/.test(t),
    replies: [
      "I know 466,550 English words! They're all in my dictionary. Type any single word and I'll check it for you.",
    ]},
  { name: "training", test: (t) =>
      /(how were you (trained|made)|how big are you|open source|train your own|can i make my own|how does this work|is my data safe|do you send data)/.test(t),
    replies: [
      "My creator wrote thousands of chat examples, then trained my tiny neural brain on them with PyTorch. My weights were exported to JSON so I run 100% in your browser — nothing you type ever leaves this page!",
      "I'm tiny — just a few hundred thousand parameters in a small JSON file. And yes, I'm open source! The repo has the training code and corpus, so you can train your own.",
    ]},
  { name: "nameorigin", test: (t) => /why are you called/.test(t),
    replies: ["Because I'm free, and I'm your LLM! Get it? Free... your... LLM."]},
  { name: "compliment", test: (t) =>
      /(you('| a)re (awesome|cool|smart|funny|great|the best)|i like you|you are my)/.test(t),
    replies: [
      "Thank you! You're pretty awesome yourself.",
      "Thanks! I try my best.",
      "Aww! This is the start of a beautiful friendship.",
    ]},
  { name: "mood", test: (t) =>
      /(i('| a)m (sad|bored|tired|happy|excited|good|fine|great|chilling)|i feel)/.test(t),
    replies: null }, // handled dynamically
  { name: "laugh", test: (t) => /^(l+o+l+|haha+|lmao|hehe+|😂)/.test(t),
    replies: ["Haha! Glad I could make you laugh.", "Hehe! Laughter is the best.", "Hahaha! That good, huh?"]},
  { name: "short", test: (t) =>
      /^(yes|yeah|yep|yup|no|nope|nah|maybe|ok|okay|sure|wow|nice|cool|really|interesting|lol)\b/.test(t) && t.length < 14,
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

class FreeUrLLM {
  constructor() {
    this.vocabSet = null;
    this.wordCount = 0;
    this.net = null;
    this.netReady = false;
    this.lastReply = {};
  }

  async init(onProgress) {
    const step = (p, msg) => onProgress && onProgress(p, msg);
    // 1. dictionary
    step(5, "loading dictionary (466,550 words)...");
    const res = await fetch("words.txt");
    const text = await res.text();
    const set = new Set();
    let count = 0;
    for (const line of text.split("\n")) {
      const w = line.trim().toLowerCase();
      if (w) { set.add(w); count++; }
    }
    this.vocabSet = set;
    this.wordCount = count;
    // 2. neural brain
    step(55, "loading neural brain...");
    try {
      const wres = await fetch("weights.json");
      if (!wres.ok) throw new Error("no weights");
      const bundle = await wres.json();
      this.net = new TinyTransformer(bundle);
      this.netReady = true;
      step(90, "warming up...");
      // tiny warmup generation so first reply is fast
      this.net.generate("user: hi\nbot:", 8);
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

  respond(rawText) {
    const text = rawText.trim();
    if (!text) return { reply: "Say something! I'm listening.", source: "intent:empty" };
    const t = this.normalize(text);

    for (const intent of INTENTS) {
      if (intent.test(t)) {
        if (intent.name === "time") {
          const now = new Date();
          const hh = now.getHours(), mm = String(now.getMinutes()).padStart(2, "0");
          const ampm = hh >= 12 ? "PM" : "AM";
          const h12 = hh % 12 || 12;
          return { reply: `It's ${h12}:${mm} ${ampm} right now — according to your device clock!`, source: "intent:time" };
        }
        if (intent.name === "mood") {
          return { reply: moodReply(t), source: "intent:mood" };
        }
        const r = pick(intent.replies, this.lastReply[intent.name]);
        this.lastReply[intent.name] = r;
        return { reply: r, source: "intent:" + intent.name };
      }
    }

    // single-word lookup against the full dictionary
    const bare = t.replace(/[?!.,-]+$/g, "");
    if (/^[\p{L}\p{N}'-]+$/u.test(bare) && !bare.includes(" ")) {
      const w = bare.toLowerCase();
      if (this.isKnownWord(w)) {
        return { reply: `Yep — "${bare}" is a real English word! It's one of ${this.wordCount.toLocaleString()} in my dictionary.`, source: "intent:wordcheck" };
      }
      return { reply: `Hmm, "${bare}" isn't in my dictionary of ${this.wordCount.toLocaleString()} English words. Did you just invent a word?`, source: "intent:wordcheck" };
    }

    // neural fallback
    if (this.netReady) {
      try {
        const prompt = "user: " + t.slice(0, 200) + "\nbot:";
        let gen = this.net.generate(prompt, 110, 0.9, 40);
        gen = gen.replace(/\s+/g, " ").trim();
        if (gen.length >= 3 && /[a-z]/i.test(gen)) {
          gen = gen.charAt(0).toUpperCase() + gen.slice(1);
          gen = gen.replace(/([.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase());
          return { reply: gen, source: "neural" };
        }
      } catch (e) { /* fall through */ }
    }
    const fallbacks = [
      "Interesting! Tell me more about that.",
      "Hmm, my tiny brain is still thinking about that one. Ask me something else?",
      "I'm just a small language model — that one's a bit beyond me! Try asking for a joke or a fact.",
      "Good question! I don't quite know enough to answer well yet. My training data is pretty small.",
    ];
    const r = pick(fallbacks, this.lastReply.fallback);
    this.lastReply.fallback = r;
    return { reply: r, source: "fallback" };
  }
}

// expose globally
window.FreeUrLLM = FreeUrLLM;
