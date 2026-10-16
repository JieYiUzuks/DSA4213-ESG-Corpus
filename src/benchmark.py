"""
benchmark.py - inference throughput for the chosen neural models, measured in ONE process.
Same workload (teacher-forced forward over the test windows), warm-up passes first, then
REPS timed rounds in which model order is shuffled each round, so drift/thermal noise
hits every model equally. Reports median and min-max tokens/sec.
Run from "Assignment 1":
    python src/benchmark.py --device mps
    python src/benchmark.py --device cpu     # same hardware class as the n-gram
"""
import argparse, json, random, statistics, time
from pathlib import Path

import torch

from data import get_loaders, PAD, SEED
from generate import load_model, CHOSEN
from train import sync

RES = Path("results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--reps", type=int, default=9)
    a = p.parse_args()
    random.seed(SEED)
    dev = torch.device(a.device)

    _, _, test_dl, itos = get_loaders(64)
    V = len(itos)
    xs = [x.to(dev) for x, _ in test_dl]
    ntok = sum((y != PAD).sum().item() for _, y in test_dl)

    models = {n: load_model(t, V, dev) for n, t in CHOSEN.items()}
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}

    with torch.no_grad():
        for m in models.values():                       # warm-up
            for _ in range(2):
                for x in xs: m(x)
        times = {n: [] for n in models}
        for _ in range(a.reps):
            order = list(models); random.shuffle(order)
            for n in order:
                sync(dev); t0 = time.perf_counter()
                for x in xs: models[n](x)
                sync(dev)
                times[n].append(ntok / (time.perf_counter() - t0))

    out = {}
    for n, v in times.items():
        out[n] = {"tag": CHOSEN[n], "params": params[n], "median_tok_s": statistics.median(v),
                  "min_tok_s": min(v), "max_tok_s": max(v)}
        print(f"{n:12s} {CHOSEN[n]:22s} params {params[n]:>9,d}  median {out[n]['median_tok_s']:>9,.0f} tok/s"
              f"  (min {min(v):,.0f}, max {max(v):,.0f})")
    out["_meta"] = {"device": a.device, "reps": a.reps, "test_tokens": ntok, "torch": torch.__version__}
    (RES / f"benchmark_{a.device}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()