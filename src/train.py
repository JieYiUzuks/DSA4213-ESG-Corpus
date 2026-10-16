"""
train.py - train one neural LM.   Run from "Assignment 1":
    python src/train.py --model lstm
    python src/train.py --model rnn
    python src/train.py --model transformer
Part II later adds:  --emb_weights path.npy --freeze_emb --tag lstm_glove
"""
import argparse, time, json, csv, random, platform
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from data import get_loaders, PAD, SEED, BLOCK_SIZE
from models import build_model

RES = Path("results")


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, dev, V):
    """Token-weighted mean cross-entropy, pad targets excluded."""
    model.eval(); tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1),
                               ignore_index=PAD, reduction="sum")
        tot += loss.item(); n += (y != PAD).sum().item()
    return tot / n


@torch.no_grad()
def throughput(model, loader, dev, reps=3):
    """Tokens/sec for a teacher-forced forward pass over the test set (1 warm-up pass)."""
    model.eval()
    ntok = sum((y != PAD).sum().item() for _, y in loader)
    for x, _ in loader: model(x.to(dev))
    sync(dev); t0 = time.perf_counter()
    for _ in range(reps):
        for x, _ in loader: model(x.to(dev))
    sync(dev)
    return ntok * reps / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["rnn", "lstm", "transformer"])
    p.add_argument("--emb", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default=None)
    p.add_argument("--emb_weights", default=None)
    p.add_argument("--freeze_emb", action="store_true")
    a = p.parse_args()

    tag = a.tag or f"{a.model}_s{a.seed}"
    RES.mkdir(exist_ok=True)
    set_seed(a.seed); dev = get_device()
    train_dl, val_dl, test_dl, itos = get_loaders(a.bs, seed=a.seed)
    V = len(itos)

    w = np.load(a.emb_weights) if a.emb_weights else None
    model = build_model(a.model, V, emb_dim=a.emb, hidden=a.hidden, layers=a.layers,
                        dropout=a.dropout, heads=a.heads, ff=a.ff, block_size=BLOCK_SIZE,
                        emb_weights=w, freeze_emb=a.freeze_emb).to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{tag} | device={dev} | params={n_all:,} (trainable {n_train:,})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    best, bad, train_time, rows = float("inf"), 0, 0.0, []

    for ep in range(1, a.epochs + 1):
        model.train(); sync(dev); t0 = time.perf_counter()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sync(dev); train_time += time.perf_counter() - t0       # update steps only

        tr = evaluate(model, train_dl, dev, V)   # eval mode (no dropout) -> comparable to val
        va = evaluate(model, val_dl, dev, V)
        rows.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        print(f"ep {ep:02d}  train {tr:.3f}  val {va:.3f}  val_ppl {np.exp(va):.1f}")
        if va < best:
            best, bad = va, 0; torch.save(model.state_dict(), RES / f"{tag}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    with open(RES / f"{tag}_log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        wr.writeheader(); wr.writerows(rows)

    model.load_state_dict(torch.load(RES / f"{tag}.pt", map_location=dev))
    te = evaluate(model, test_dl, dev, V)
    tps = throughput(model, test_dl, dev)
    result = {
        "tag": tag, "model": a.model, "args": vars(a),
        "best_val_loss": best, "best_val_ppl": float(np.exp(best)),
        "test_loss": te, "test_ppl": float(np.exp(te)),
        "params_total": n_all, "params_trainable": n_train,
        "train_time_s": train_time, "epochs_run": len(rows),
        "best_epoch": int(np.argmin([r["val_loss"] for r in rows]) + 1),
        "infer_tokens_per_s": tps,
        "hardware": f"{platform.platform()} | {platform.processor()} | torch {torch.__version__} | {dev}",
    }
    (RES / f"{tag}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("test_loss", "test_ppl", "params_total",
                                              "train_time_s", "infer_tokens_per_s")}, indent=2))


if __name__ == "__main__":
    main()
