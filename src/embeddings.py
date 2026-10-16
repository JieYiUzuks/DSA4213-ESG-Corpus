"""
embeddings.py - build the two FROZEN embedding tables for Part II.
Run from "Assignment 1":
    python src/embeddings.py sgns    # own Skip-gram w/ negative sampling, TRAIN split only
    python src/embeddings.py glove   # public GloVe 6B 100d (data/glove/glove.6B.100d.txt)
Outputs: data/processed/emb_sgns100.npy, emb_glove100.npy  (shape V x 100, row i = itos[i])
         results/sgns_cost.json, results/glove_coverage.json

Both tables are multiplied by ONE scalar so their overall std = 1 (the default std of
nn.Embedding's random init), keeping the scale comparable across the three settings.
<pad> row is zeros in both tables. Missing GloVe words / <unk> / <eos> are drawn from a
normal with the per-dimension mean and std of the GloVe vectors that were found (seeded).
"""
import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from data import prepare, PAD, SEED
from train import get_device, sync

PROC, RES = Path("data/processed"), Path("results")
GLOVE_PATH = Path("data/glove/glove.6B.100d.txt")
DIM = 100
PROBE = ["emissions", "board", "energy", "safety", "employees", "climate", "risk"]


def neighbours(table, itos, words=PROBE, k=5):
    stoi = {w: i for i, w in enumerate(itos)}
    T = table / (np.linalg.norm(table, axis=1, keepdims=True) + 1e-9)
    for w in words:
        if w not in stoi:
            continue
        sims = T @ T[stoi[w]]
        top = [itos[j] for j in np.argsort(-sims)[1:k + 1]]
        print(f"  {w:10s} -> {', '.join(top)}")


def rescale(table, mask):
    """Multiply the whole table by one scalar so the std of rows in `mask` is 1."""
    s = 1.0 / table[mask].std()
    return (table * s).astype(np.float32), float(s)


# ---------------------------------------------------------------- Skip-gram (SGNS)
def build_pairs(ids, window):
    ids = torch.tensor(ids)
    cs, xs = [], []
    for o in range(1, window + 1):
        a, b = ids[:-o], ids[o:]
        cs += [a, b]; xs += [b, a]
    return torch.cat(cs), torch.cat(xs)


def run_sgns(a):
    torch.manual_seed(SEED); np.random.seed(SEED)
    data, itos = prepare(); V = len(itos)
    ids = data["train"]                                   # TRAIN split only
    dev = get_device()

    counts = torch.bincount(torch.tensor(ids), minlength=V).float()
    freq = (counts / counts.sum()).clamp_min(1e-12)
    keep = ((torch.sqrt(freq / a.sub_t) + 1) * a.sub_t / freq).clamp(max=1.0)   # word2vec subsampling
    noise = counts ** 0.75; noise /= noise.sum()                                 # unigram^0.75 (CPU)

    cen, ctx = build_pairs(ids, a.window)
    inn, out = nn.Embedding(V, DIM).to(dev), nn.Embedding(V, DIM).to(dev)
    nn.init.uniform_(inn.weight, -0.5 / DIM, 0.5 / DIM); nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inn.parameters()) + list(out.parameters()), lr=a.lr)

    sync(dev); t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        m = torch.rand(len(cen)) < keep[cen] * keep[ctx]
        c, x = cen[m], ctx[m]
        perm = torch.randperm(len(c)); c, x = c[perm], x[perm]
        tot, nb = 0.0, 0
        for i in range(0, len(c), a.bs):
            cb, xb = c[i:i + a.bs].to(dev), x[i:i + a.bs].to(dev)
            neg = torch.multinomial(noise, len(cb) * a.neg, replacement=True).view(len(cb), a.neg).to(dev)
            v = inn(cb)
            pos = (v * out(xb)).sum(-1)
            ng = torch.bmm(out(neg), v.unsqueeze(-1)).squeeze(-1)
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-ng).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep:02d}  pairs {len(c):,d}  loss {tot / nb:.4f}")
    sync(dev); cost = time.perf_counter() - t0

    table = inn.weight.detach().cpu().numpy().copy()
    table[PAD] = 0.0
    mask = np.arange(V) != PAD
    table, scale = rescale(table, mask)
    PROC.mkdir(parents=True, exist_ok=True); RES.mkdir(exist_ok=True)
    np.save(PROC / "emb_sgns100.npy", table)
    (RES / "sgns_cost.json").write_text(json.dumps({
        "train_time_s": cost, "device": str(dev), "epochs": a.epochs, "window": a.window,
        "neg": a.neg, "lr": a.lr, "batch": a.bs, "subsample_t": a.sub_t, "dim": DIM,
        "rescale_factor": scale, "train_tokens": len(ids)}, indent=2))
    print(f"\nSGNS trained in {cost:.1f}s on {dev}; saved emb_sgns100.npy (rescale x{scale:.2f})")
    print("Nearest neighbours (cosine):"); neighbours(table, itos)


# ---------------------------------------------------------------- GloVe
def run_glove(a):
    data, itos = prepare(); V = len(itos)
    need = set(itos[3:])                                  # real vocab (incl. <num>), not specials
    found = {}
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            w, rest = line.rstrip("\n").split(" ", 1)
            if w in need:
                found[w] = np.asarray(rest.split(), dtype=np.float32)
    covered = np.stack(list(found.values()))
    mu, sd = covered.mean(0), covered.std(0)

    rng = np.random.default_rng(SEED)
    table = np.zeros((V, DIM), dtype=np.float32)
    n_rand = 0
    for i, w in enumerate(itos):
        if i == PAD:
            continue
        if w in found:
            table[i] = found[w]
        else:
            table[i] = rng.normal(mu, sd); n_rand += 1
    mask = np.arange(V) != PAD
    table, scale = rescale(table, mask)

    train = data["train"]
    tok_counts = Counter(train)
    real = [i for i in range(3, V)]
    tok_total = sum(tok_counts[i] for i in real)
    tok_cov = sum(tok_counts[i] for i in real if itos[i] in found)
    missing = sorted([i for i in real if itos[i] not in found], key=lambda i: -tok_counts[i])[:25]
    cov = {
        "vocab_real_types": len(real), "found_in_glove": len(found),
        "type_coverage": len(found) / len(real),
        "train_token_coverage_excl_specials": tok_cov / tok_total,
        "randomly_initialised_rows_incl_unk_eos": n_rand,
        "pad_row": "zeros",
        "init_rule": "Normal(per-dim mean, per-dim std of found GloVe vectors), seed %d" % SEED,
        "rescale_factor": scale,
        "top_missing_words": [(itos[i], tok_counts[i]) for i in missing],
    }
    PROC.mkdir(parents=True, exist_ok=True); RES.mkdir(exist_ok=True)
    np.save(PROC / "emb_glove100.npy", table)
    (RES / "glove_coverage.json").write_text(json.dumps(cov, indent=2))
    print(f"GloVe found {len(found):,d}/{len(real):,d} vocab types ({cov['type_coverage']:.1%}); "
          f"covers {cov['train_token_coverage_excl_specials']:.1%} of train tokens")
    print("Most frequent missing:", cov["top_missing_words"][:15])
    print("Nearest neighbours (cosine):"); neighbours(table, itos)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sgns", "glove"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--neg", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--sub_t", type=float, default=1e-3)
    a = p.parse_args()
    run_sgns(a) if a.mode == "sgns" else run_glove(a)