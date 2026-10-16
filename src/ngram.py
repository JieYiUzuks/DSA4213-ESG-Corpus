"""
ngram.py - trigram LM with linear interpolation (unigram add-one + bigram + trigram).
Interpolation weights are tuned on the VALIDATION set only. Evaluated on exactly the same
64-token windows / targets as the neural models (context is reset at each window start;
pad targets excluded). If a context was never seen, its weight is redistributed to the
lower orders so the distribution stays proper.
Run from "Assignment 1":  python src/ngram.py
"""
import json, time, itertools
from collections import Counter
from pathlib import Path

import numpy as np
from data import prepare, WindowDataset, PAD, BLOCK_SIZE

RES = Path("results")


class TrigramLM:
    def __init__(self, ids, V):
        self.V, self.N = V, len(ids)
        self.uni = Counter(ids)
        self.bi = Counter(zip(ids, ids[1:]))
        self.tri = Counter(zip(ids, ids[1:], ids[2:]))
        self.ctx1 = Counter(ids[:-1])                          # c(w1) as a bigram context
        self.ctx2 = Counter(zip(ids[:-2], ids[1:-1]))          # c(w1,w2) as a trigram context

    def size(self):
        return len(self.uni) + len(self.bi) + len(self.tri)

    def components(self, w, h1, h2):
        """-> (3 probabilities, 3 availability flags) for unigram, bigram, trigram."""
        p_uni = (self.uni[w] + 1) / (self.N + self.V)
        c1 = self.ctx1[h1]
        p_bi = self.bi[(h1, w)] / c1 if c1 else 0.0
        c2 = self.ctx2[(h2, h1)] if h2 is not None else 0
        p_tri = self.tri[(h2, h1, w)] / c2 if c2 else 0.0
        return (p_uni, p_bi, p_tri), (True, c1 > 0, c2 > 0)


def collect(lm, ids):
    """Component probabilities for every non-pad target in the windowed split."""
    ds = WindowDataset(ids, BLOCK_SIZE)
    P, A = [], []
    for w in ds.windows.tolist():
        x, y = w[:-1], w[1:]
        for t in range(len(y)):
            if y[t] == PAD: break
            p, a = lm.components(y[t], x[t], x[t - 1] if t >= 1 else None)
            P.append(p); A.append(a)
    return np.array(P), np.array(A, dtype=float)


def avg_nll(P, A, lam):
    w = A * np.asarray(lam)
    dead = w.sum(1) == 0            # no usable order has weight -> fall back to unigram
    w[dead, 0] = 1.0
    w = w / w.sum(1, keepdims=True)
    return float(-np.log((w * P).sum(1)).mean())


def main():
    data, itos = prepare()
    V = len(itos)
    t0 = time.perf_counter()
    lm = TrigramLM(data["train"], V)
    train_time = time.perf_counter() - t0                    # "training" = counting

    Pv, Av = collect(lm, data["val"])
    grid = [(a / 10, b / 10, (10 - a - b) / 10) for a in range(0, 11) for b in range(0, 11 - a)]
    best = min(grid, key=lambda l: avg_nll(Pv, Av, l))
    val_ce = avg_nll(Pv, Av, best)

    t0 = time.perf_counter()
    Pt, At = collect(lm, data["test"])                       # inference workload
    test_ce = avg_nll(Pt, At, best)
    infer_time = time.perf_counter() - t0

    res = {"tag": "ngram", "lambdas_uni_bi_tri": best,
           "val_loss": val_ce, "val_ppl": float(np.exp(val_ce)),
           "test_loss": test_ce, "test_ppl": float(np.exp(test_ce)),
           "stored_ngrams": lm.size(), "train_time_s": train_time,
           "infer_tokens_per_s": len(Pt) / infer_time}
    RES.mkdir(exist_ok=True)
    (RES / "ngram_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()