"""
generate.py - matched-decoding generation for n-gram, RNN, LSTM, Transformer.
Run from "Assignment 1":  python src/generate.py
Checkpoints come from results/chosen.json (written by select_best.py, VALIDATION perplexity).
    python src/generate.py --set part1   |   --set part2

Matched settings for every model: same prompts, temperature, top-k, generation length,
per-prompt random seed, <pad> never sampled, context cropped to the last BLOCK_SIZE tokens
(the n-gram only uses its last 2 tokens by construction).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data import prepare, tokenize_paragraph, BLOCK_SIZE, PAD, UNK, SEED
from models import build_model
from ngram import TrigramLM

RES = Path("results")
def _chosen():
    """Checkpoints picked on VALIDATION perplexity by select_best.py (results/chosen.json)."""
    f = RES / "chosen.json"
    c = json.loads(f.read_text()) if f.exists() else {}
    return {"rnn": c.get("rnn", "rnn_lr1e-3"), "lstm": c.get("lstm", "lstm_lr3e-3"),
            "transformer": c.get("transformer", "transformer_lr3e-3")}


_C = _chosen()
# Part I: one checkpoint per architecture (also imported by benchmark.py / classify.py)
CHOSEN = {"RNN": _C["rnn"], "LSTM": _C["lstm"], "Transformer": _C["transformer"]}
# Part II: the chosen LSTM with different input-embedding settings (frozen = required comparison;
# *_init = same tables used only as initialisation, then trained: separates trainability from data)
CHOSEN_PART2 = {"LSTM random-trainable": _C["lstm"],
                "LSTM own-SGNS-frozen": "lstm_sgns_frozen",
                "LSTM GloVe-frozen": "lstm_glove_frozen",
                "LSTM own-SGNS-init-trainable": "lstm_sgns_init",
                "LSTM GloVe-init-trainable": "lstm_glove_init"}
PROMPTS = [
    "our greenhouse gas emissions",
    "we are committed to",
    "the board of directors",
    "in 2025 , we",
    "renewable energy capacity",
]
GEN_LEN, TEMP, TOPK = 40, 0.8, 40       # shown generations
N_STAT, STAT_LEN = 30, 60               # samples used for repetition statistics


def load_model(tag, V, dev="cpu"):
    """Rebuild a trained neural LM from results/{tag}_result.json + results/{tag}.pt."""
    a = json.loads((RES / f"{tag}_result.json").read_text())["args"]
    m = build_model(a["model"], V, emb_dim=a["emb"], hidden=a["hidden"], layers=a["layers"],
                    dropout=a["dropout"], heads=a["heads"], ff=a["ff"], block_size=BLOCK_SIZE)
    m.load_state_dict(torch.load(RES / f"{tag}.pt", map_location=dev))
    return m.to(dev).eval()


class NgramSampler:
    """Same interpolation (and same renormalisation over available orders) as ngram.py."""
    def __init__(self, train_ids, V):
        self.lm, self.V = TrigramLM(train_ids, V), V
        self.lam = json.loads((RES / "ngram_result.json").read_text())["lambdas_uni_bi_tri"]
        self.pu = np.array([(self.lm.uni[w] + 1) / (self.lm.N + V) for w in range(V)])
        self.bn, self.tn = defaultdict(dict), defaultdict(dict)
        for (a, b), c in self.lm.bi.items(): self.bn[a][b] = c
        for (a, b, c_), c in self.lm.tri.items(): self.tn[(a, b)][c_] = c

    def probs(self, ids):
        lm, V = self.lm, self.V
        h1 = ids[-1]; h2 = ids[-2] if len(ids) > 1 else None
        parts = [(self.lam[0], self.pu)]
        c1 = lm.ctx1[h1]
        if c1:
            pb = np.zeros(V)
            for w, c in self.bn[h1].items(): pb[w] = c / c1
            parts.append((self.lam[1], pb))
        c2 = lm.ctx2[(h2, h1)] if h2 is not None else 0
        if c2:
            pt = np.zeros(V)
            for w, c in self.tn[(h2, h1)].items(): pt[w] = c / c2
            parts.append((self.lam[2], pt))
        tot = sum(l for l, _ in parts)
        if tot == 0: return self.pu
        return sum(l * v for l, v in parts) / tot

    def step_fn(self):
        return lambda ids: torch.tensor(np.log(self.probs(ids) + 1e-30), dtype=torch.float)


def neural_step(model):
    @torch.no_grad()
    def f(ids):
        return model(torch.tensor([ids[-BLOCK_SIZE:]]))[0, -1]
    return f


def pick(logits, gen):
    logits = logits.clone().float()
    logits[PAD] = -float("inf")
    logits = logits / TEMP
    kth = torch.topk(logits, TOPK).values[-1]
    logits[logits < kth] = -float("inf")
    return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))


def generate(step_fn, prompt_ids, n, seed):
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids)
    for _ in range(n):
        ids.append(pick(step_fn(ids), gen))
    return ids[len(prompt_ids):]


def rep_stats(samples):
    """Per-sample: distinct-2 (unique bigrams / bigrams), repeated-trigram rate, <unk> rate."""
    d2, rt, uk = [], [], []
    for s in samples:
        bi = list(zip(s, s[1:])); tri = list(zip(s, s[1:], s[2:]))
        d2.append(len(set(bi)) / len(bi))
        seen, rep = set(), 0
        for t in tri:
            rep += t in seen; seen.add(t)
        rt.append(rep / len(tri))
        uk.append(sum(i == UNK for i in s) / len(s))
    return {"distinct2": float(np.mean(d2)), "repeated_trigram_rate": float(np.mean(rt)),
            "unk_rate": float(np.mean(uk))}


def main():
    data, itos = prepare()
    V = len(itos); stoi = {w: i for i, w in enumerate(itos)}
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["part1", "part2"], default="part1")
    args = ap.parse_args()
    models = {}
    if args.set == "part1":                       # n-gram only appears in the Part I comparison
        models["n-gram"] = NgramSampler(data["train"], V).step_fn()
    chosen = CHOSEN if args.set == "part1" else CHOSEN_PART2
    for name, tag in chosen.items():
        if not (RES / f"{tag}_result.json").exists():
            print(f"[skip] {name}: no checkpoint {tag}"); continue
        models[f"{name} ({tag})"] = neural_step(load_model(tag, V))

    enc = lambda text: [stoi.get(t, UNK) for t in tokenize_paragraph(text)]
    dec = lambda ids: " ".join(itos[i] for i in ids)

    out = [f"# Generations (temp={TEMP}, top-k={TOPK}, {GEN_LEN} tokens, seed={SEED}+prompt index)\n"]
    for pi, prompt in enumerate(PROMPTS):
        out.append(f"\n## Prompt: {prompt}\n")
        for name, fn in models.items():
            g = generate(fn, enc(prompt), GEN_LEN, SEED + pi)
            out.append(f"- **{name}**: {prompt} | {dec(g)}")
    text = "\n".join(out)
    print(text)
    RES.mkdir(exist_ok=True)
    (RES / f"generations_{args.set}.md").write_text(text)

    stats = {}
    for name, fn in models.items():
        samples = [generate(fn, enc(PROMPTS[i % len(PROMPTS)]), STAT_LEN, SEED + 1000 + i)
                   for i in range(N_STAT)]
        stats[name] = rep_stats(samples)
    print("\n## Repetition statistics (", N_STAT, "samples x", STAT_LEN, "tokens )")
    for k, v in stats.items():
        print(f"{k:38s} distinct-2 {v['distinct2']:.3f}  repeated-trigram {v['repeated_trigram_rate']:.3f}  unk {v['unk_rate']:.3f}")
    (RES / f"generation_stats_{args.set}.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
