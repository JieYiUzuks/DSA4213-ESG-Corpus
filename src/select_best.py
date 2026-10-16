"""
select_best.py - choose the learning rate per architecture by VALIDATION perplexity (seed 42 runs).
    python src/select_best.py            # prints the grid, writes results/chosen.json
    python src/select_best.py --get lstm # prints the chosen tag   (used by run_all.sh)
    python src/select_best.py --lr lstm  # prints the chosen learning rate
Only runs named {rnn|lstm|transformer}_lr<value> (no seed suffix) take part in the grid.
Test perplexity is never used for selection.
"""
import argparse, json, re, sys
from pathlib import Path

RES = Path("results")
PAT = re.compile(r"^(rnn|lstm|transformer)_lr[0-9.e-]+$")


def grid():
    rows = []
    for f in sorted(RES.glob("*_result.json")):
        r = json.loads(f.read_text())
        if PAT.match(r["tag"]):
            rows.append({"model": r["model"], "tag": r["tag"], "lr": r["args"]["lr"],
                         "val_ppl": r["best_val_ppl"], "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"], "epochs_cap": r["args"]["epochs"]})
    return rows


def chosen(rows):
    best = {}
    for r in rows:
        if r["model"] not in best or r["val_ppl"] < best[r["model"]]["val_ppl"]:
            best[r["model"]] = r
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--get"); ap.add_argument("--lr")
    a = ap.parse_args()
    rows = grid(); best = chosen(rows)
    if a.get or a.lr:
        m = a.get or a.lr
        if m not in best:
            sys.exit(f"no finished grid runs for {m}")
        print(best[m]["tag"] if a.get else best[m]["lr"])
        sys.exit(0)
    print(f"{'model':12s} {'tag':22s} {'lr':>7s} {'val ppl':>8s} {'best ep':>8s} {'epochs run/cap':>15s}  chosen")
    for r in sorted(rows, key=lambda r: (r["model"], r["lr"])):
        star = "  <--" if best[r["model"]]["tag"] == r["tag"] else ""
        print(f"{r['model']:12s} {r['tag']:22s} {r['lr']:>7g} {r['val_ppl']:>8.2f} {r['best_epoch']:>8d} "
              f"{str(r['epochs_run']) + '/' + str(r['epochs_cap']):>15s}{star}")
        if r["epochs_run"] >= r["epochs_cap"]:
            print(f"{'':12s} WARNING: {r['tag']} hit its epoch cap (not converged) - its val ppl is a lower bound on quality")
    RES.mkdir(exist_ok=True)
    (RES / "chosen.json").write_text(json.dumps({m: r["tag"] for m, r in best.items()}, indent=2))
    print("\nwrote results/chosen.json:", {m: r["tag"] for m, r in best.items()})
