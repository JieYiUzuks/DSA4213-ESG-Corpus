"""
extra_checks.py - evidence for correctness and for the "why" behind Part I results.
Run from "Assignment 1":
    python src/extra_checks.py tests      # causality / padding / loss-masking tests (random weights)
    python src/extra_checks.py params     # embedding / core / output parameter breakdown (trained models)
    python src/extra_checks.py position   # test loss as a function of position in the 64-token window
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import prepare, WindowDataset, BLOCK_SIZE, PAD
from models import build_model

RES = Path("results")


# ------------------------------------------------------------------ 1. tests
def run_tests():
    torch.manual_seed(0)
    V, B, T = 200, 4, BLOCK_SIZE
    ok = True
    for name in ["rnn", "lstm", "transformer"]:
        m = build_model(name, V, emb_dim=100, hidden=256, dropout=0.3, block_size=T).eval()
        x = torch.randint(3, V, (B, T))
        with torch.no_grad():
            base = m(x)
            for t in [0, 5, 31, 62]:
                # (1) CAUSALITY: changing tokens AFTER position t must not change logits at <= t
                x2 = x.clone(); x2[:, t + 1:] = torch.randint(3, V, (B, T - t - 1))
                same = torch.allclose(base[:, :t + 1], m(x2)[:, :t + 1], atol=1e-5)
                # (2) PADDING: trailing <pad> must not change logits of the real tokens
                x3 = x.clone(); x3[:, t + 1:] = PAD
                same_pad = torch.allclose(base[:, :t + 1], m(x3)[:, :t + 1], atol=1e-5)
                # (3) NON-TRIVIAL: changing the PREVIOUS token must change this position's logits
                #     (otherwise the model would be ignoring its context altogether)
                if t >= 1:
                    x4 = x.clone(); x4[:, t - 1] = (x4[:, t - 1] - 3 + 1) % (V - 3) + 3
                    uses_past = not torch.allclose(base[:, t], m(x4)[:, t], atol=1e-5)
                else:
                    uses_past = True
                good = same and same_pad and uses_past
                ok &= good
                print(f"{name:12s} t={t:2d}  causal={same}  pad-invariant={same_pad}  uses_past={uses_past}"
                      f"  -> {'PASS' if good else 'FAIL'}")

    # (4) loss masking: padded targets must not contribute
    logits = torch.randn(2, 6, V)
    y = torch.randint(3, V, (2, 6)); y[:, 4:] = PAD
    a = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD)
    b = F.cross_entropy(logits[:, :4].reshape(-1, V), y[:, :4].reshape(-1))
    good = torch.allclose(a, b, atol=1e-6)
    ok &= good
    print(f"loss ignores <pad> targets -> {'PASS' if good else 'FAIL'}")
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")


# ------------------------------------------------------------------ 2. parameter breakdown
def run_params():
    from generate import load_model, CHOSEN, CHOSEN_PART2
    _, itos = prepare(); V = len(itos)
    rows = []
    for tag in dict.fromkeys(list(CHOSEN.values()) + list(CHOSEN_PART2.values())):
        m = load_model(tag, V)
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        emb, out, total = n(m.emb), n(m.out), n(m)
        trainable = json.loads((RES / f"{tag}_result.json").read_text())["params_trainable"]  # freeze is applied at train time
        rows.append({"tag": tag, "total": total, "trainable": trainable, "input_embedding": emb,
                     "core_sequence_layers": total - emb - out, "output_layer": out})
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RES / "param_breakdown.csv", index=False)


# ------------------------------------------------------------------ 3. loss by position
@torch.no_grad()
def neural_by_position(tag, windows, V):
    from generate import load_model
    m = load_model(tag, V)
    sums, cnt = torch.zeros(BLOCK_SIZE), torch.zeros(BLOCK_SIZE)
    for i in range(0, len(windows), 128):
        w = windows[i:i + 128]; x, y = w[:, :-1], w[:, 1:]
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1), ignore_index=PAD,
                               reduction="none").view(y.shape)
        mask = (y != PAD).float()
        sums += (loss * mask).sum(0); cnt += mask.sum(0)
    return (sums / cnt.clamp_min(1)).numpy()


def ngram_by_position(data, V, windows):
    from ngram import TrigramLM
    lm = TrigramLM(data["train"], V)
    lam = np.array(json.loads((RES / "ngram_result.json").read_text())["lambdas_uni_bi_tri"])
    sums, cnt = np.zeros(BLOCK_SIZE), np.zeros(BLOCK_SIZE)
    for w in windows.tolist():
        x, y = w[:-1], w[1:]
        for t in range(len(y)):
            if y[t] == PAD: break
            p, a = lm.components(y[t], x[t], x[t - 1] if t >= 1 else None)
            wts = np.array(a, float) * lam
            if wts.sum() == 0: wts[0] = 1.0
            wts /= wts.sum()
            sums[t] += -np.log((wts * np.array(p)).sum()); cnt[t] += 1
    return sums / np.maximum(cnt, 1)


def run_position():
    from generate import CHOSEN
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, itos = prepare(); V = len(itos)
    windows = WindowDataset(data["test"], BLOCK_SIZE).windows
    curves = {"n-gram": ngram_by_position(data, V, windows)}
    for name, tag in CHOSEN.items():
        curves[name] = neural_by_position(tag, windows, V)
    bins = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    table = pd.DataFrame({k: [v[a - 1:b].mean() for a, b in bins] for k, v in curves.items()},
                         index=[f"pos {a}-{b}" for a, b in bins]).round(3)
    print("mean test cross-entropy (nats/token) by position in window:\n", table.to_string())
    table.to_csv(RES / "loss_by_position.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    k = 4                                                     # light smoothing for readability
    for name, v in curves.items():
        sm = np.convolve(v, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + (k + 1) / 2, sm, label=name)
    ax.set_xlabel("position in 64-token window"); ax.set_ylabel("test cross-entropy (nats/token)")
    ax.legend(); plt.tight_layout(); plt.savefig(RES / "fig_loss_by_position.png", dpi=200)
    print("saved results/fig_loss_by_position.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"tests": run_tests, "params": run_params, "position": run_position}.get(
        cmd, lambda: print(__doc__))()
