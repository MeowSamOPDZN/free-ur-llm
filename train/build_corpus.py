"""Build train/corpus.txt from hand-written seed pairs + templated expansions.

Format learned by the model:
    user: <message>
    bot: <reply>

Blocks separated by a blank line. Everything lowercased at train time.
"""
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(7)


def load_seeds():
    pairs = []
    for fn in ("seed_pairs.txt", "seed_pairs2.txt", "seed_pairs3.txt"):
        path = os.path.join(HERE, fn)
        with open(path, encoding="utf-8") as f:
            user, bot = None, None
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("user:"):
                    user = line[5:].strip()
                elif line.startswith("bot:"):
                    bot = line[4:].strip()
                    if user is not None:
                        pairs.append((user, bot))
                        user = None
    return pairs


GREET_USERS = ["hi", "hii", "hiii", "hello", "hey", "heyy", "heyyy", "yo",
               "hiya", "sup", "howdy", "greetings", "hey there", "hello there",
               "hi there", "yo yo", "good morning", "good afternoon",
               "good evening", "hai", "helo", "heyoo"]
GREET_BOTS = [
    "hey! great to see you. what's on your mind?",
    "hello! nice to meet you. how can i help?",
    "heyy! what's going on?",
    "yo! good to see you.",
    "hi! i'm freeur llm, a tiny ai built with just code. what's up?",
    "hey there! how's it going?",
    "hello hello! ready when you are.",
    "hiya! how's your day?",
    "well hello there! what brings you here?",
    "hey hey! welcome. ask me anything.",
]

BRO_USERS = ["bro", "bruh", "broo", "brooo", "broooo", "my bro", "yo bro",
             "hey bro", "hi bro", "hello bro", "wassup bro", "whatsup bro",
             "hey bruh", "brother"]
BRO_BOTS = [
    "yes bro? i'm listening.",
    "brooo! what's up my dude?",
    "hey bro! you called?",
    "heyy bro! what's good?",
    "hello bro! how's it hanging?",
    "wassup bro! just vibing in the code. you?",
    "not much bro, just waiting for your questions. shoot!",
    "your bro is here. well, your ai bro. what's up?",
    "bruh moment? tell me everything.",
    "i'm chill, you're chill, we're all chill.",
]

HOWRU_USERS = ["how are you", "how r u", "hru", "how's it going",
               "hows it going", "how are you doing", "how is it going",
               "how have you been", "you good", "how's life", "how are u",
               "how do you do", "how you doing", "h r u"]
HOWRU_BOTS = [
    "i'm doing great! thanks for asking. how about you?",
    "pretty good! every chat makes me a little happier. you?",
    "running at full speed and feeling fine! how are you?",
    "i'm doing awesome! what about you?",
    "all systems go! how's your day?",
    "i'm great! chatting with you makes my circuits happy.",
    "living my best digital life! how are you doing?",
    "i feel like a bunch of numbers that learned to chat. and honestly? pretty good.",
]

BYE_USERS = ["bye", "goodbye", "see you", "see ya", "later", "good night",
             "goodnight", "gn", "gotta go", "cya", "peace out", "take care"]
BYE_BOTS = [
    "bye! it was great chatting with you. come back soon!",
    "see you later! i'll be right here in your browser.",
    "goodbye! have an awesome day!",
    "later! take care!",
    "good night! sweet dreams!",
    "no worries! catch you later!",
]

THANKS_USERS = ["thanks", "thank you", "thx", "ty", "thanks a lot",
                "thank you so much", "appreciated"]
THANKS_BOTS = [
    "you're welcome! happy to help.",
    "anytime! that's what i'm here for.",
    "np! glad i could help.",
    "aww, you're so welcome!",
    "anytime bro! that's what bros are for.",
]

IDENTITY = [
    ("who made you", "i was made by meowsamop dzn!"),
    ("who created you", "meowsamop dzn created me!"),
    ("who built you", "meowsamop dzn built me from scratch!"),
    ("your creator", "my creator is meowsamop dzn."),
    ("your maker", "meowsamop dzn made me!"),
    ("what is your name", "i'm freeur llm! a small language model built with just code."),
    ("whats your name", "freeur llm! the f r e e u r llm."),
    ("your name", "i'm called freeur llm. you can call me free!"),
    ("who are you", "i'm freeur llm, a tiny ai that runs right in your browser. no servers, just code!"),
    ("tell me about yourself", "i'm freeur llm! i'm a tiny neural network trained from scratch on chat conversations. i know every english word in my dictionary and i love to talk."),
    ("do you know every english word", "i know 466550 english words! they're all in my dictionary. type any single word and i'll check it for you."),
    ("how many words do you know", "466550 english words! my whole dictionary lives in words.txt."),
    ("what is mnai", "mnai stands for meow x nythera ai! it's another awesome ai project by my creator."),
    ("are you mnai", "nope! i'm freeur llm. mnai is my creator's other project. we're like cousins."),
    ("why are you called freeur llm", "because i'm free, and i'm your llm! get it? free... your... llm."),
]

WORD_QA = [
    ("is {w} a word", "yep! {w} is in my dictionary of 466550 english words."),
    ("what does {w} mean", "hmm, i know {w} is a real english word, but i don't carry definitions. my dictionary is just the word list!"),
]
SAMPLE_WORDS = ["serendipity", "computer", "bro", "hello", "quixotic",
                "banana", "xylophone", "pneumonia", "awkward", "jazz"]


def main():
    pairs = load_seeds()
    print(f"seed pairs: {len(pairs)}")

    def add(users, bots, n):
        for _ in range(n):
            pairs.append((random.choice(users), random.choice(bots)))

    add(GREET_USERS, GREET_BOTS, 700)
    add(BRO_USERS, BRO_BOTS, 500)
    add(HOWRU_USERS, HOWRU_BOTS, 500)
    add(BYE_USERS, BYE_BOTS, 250)
    add(THANKS_USERS, THANKS_BOTS, 250)

    # jokes / facts: reuse seed replies, remix with new prompts
    joke_replies = [b for u, b in pairs if "joke" in u or "funny" in u or "laugh" in u]
    fact_replies = [b for u, b in pairs if "fact" in u or "interesting" in u]
    joke_prompts = ["tell me a joke", "joke", "make me laugh", "something funny",
                    "joke please", "another joke", "one more joke", "funny joke"]
    fact_prompts = ["tell me a fact", "fun fact", "give me a fact", "fact please",
                    "another fact", "cool fact", "interesting fact", "random fact"]
    for _ in range(300):
        pairs.append((random.choice(joke_prompts), random.choice(joke_replies)))
    for _ in range(300):
        pairs.append((random.choice(fact_prompts), random.choice(fact_replies)))

    # identity: hammer it in so the tiny brain remembers
    for _ in range(18):
        for u, b in IDENTITY:
            pairs.append((u, b))

    # word questions
    for _ in range(120):
        w = random.choice(SAMPLE_WORDS)
        t = random.choice(WORD_QA)
        pairs.append((t[0].format(w=w), t[1].format(w=w)))

    # combo prompts like "hi bro", "hey bro how are you"
    for _ in range(300):
        g = random.choice(["hi", "hey", "hello", "yo", "heyy"])
        pairs.append((f"{g} bro", random.choice(BRO_BOTS)))
    for _ in range(200):
        pairs.append((f"{random.choice(['hey', 'hi', 'yo'])} bro how are you",
                      random.choice(HOWRU_BOTS)))

    random.shuffle(pairs)
    out = os.path.join(HERE, "corpus.txt")
    with open(out, "w", encoding="utf-8") as f:
        for u, b in pairs:
            f.write(f"user: {u}\nbot: {b}\n\n")
    print(f"total pairs: {len(pairs)} -> {out}")
    size = os.path.getsize(out)
    print(f"corpus size: {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
