"""
make_report_assets.py - build the report's tables and figures from results/.
Run from "Assignment 1" after run_all.sh:   python src/make_report_assets.py
Writes: results/table_part1.{csv,md}, table_part2.{csv,md}, table_part3.md,
        results/fig_part1_curves.png, fig_part2_curves.png, fig_part3_f1.png
Tables report mean [min-max] over the 3 training seeds (42/43/44); n-gram is deterministic (1 run).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate import CHOSEN, CHOSEN_PART2

RES = Path("results")
SEEDS = ["", "_s43", "_s44"]
# validated categorical slots 1-3 (blue, orange, aqua); light surface
BLUE, ORANGE, AQUA, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9, "legend.frameon": False})


def runs(base):
    out = []
    for s in SEEDS:
        f = RES / f"{base}{s}_result.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def ms(vals, f="{:.1f}"):
    v = np.asarray(vals, float)
    if len(v) == 1:
        return f.format(v[0])
    return f"{f.format(v.mean())} [{f.format(v.min())}-{f.format(v.max())}]"


def gap_at_best(tag):
    f = RES / f"{tag}_log.csv"
    if not f.exists():
        return np.nan
    d = pd.read_csv(f)
    r = d.loc[d.val_loss.idxmin()]
    return r.val_loss - r.train_loss          # nats; larger = more overfit


def bench(dev):
    f = RES / f"benchmark_{dev}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_table(df, name, caption):
    df.to_csv(RES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)
    (RES / f"{name}.md").write_text(f"{caption}\n\n{md}\n")
    print(f"\n{caption}\n{md}")


# ---------------------------------------------------------------- Part I table
def part1_table():
    rows = []
    ng = RES / "ngram_result.json"
    if ng.exists():
        r = json.loads(ng.read_text())
        rows.append({"model": "trigram (interpolated)", "val ppl": f"{r['val_ppl']:.1f}", "test ppl": f"{r['test_ppl']:.1f}",
                     "params / entries": f"{r['stored_ngrams']:,} n-grams", "best epoch": "-", "train time (s)": f"{r['train_time_s']:.1f}",
                     "tokens/s CPU": f"{r['infer_tokens_per_s']:,.0f}*", "tokens/s MPS": "-"})
    bc, bm = bench("cpu"), bench("mps")
    for name, tag in CHOSEN.items():
        rs = runs(tag)
        if not rs:
            continue
        rows.append({"model": f"{name} ({tag})", "val ppl": ms([r["best_val_ppl"] for r in rs]),
                     "test ppl": ms([r["test_ppl"] for r in rs]),
                     "params / entries": f"{rs[0]['params_total']:,}",
                     "best epoch": ms([r["best_epoch"] for r in rs], "{:.0f}"),
                     "train time (s)": ms([r["train_time_s"] for r in rs], "{:.0f}"),
                     "tokens/s CPU": f"{bc[name]['median_tok_s']:,.0f}" if name in bc else "-",
                     "tokens/s MPS": f"{bm[name]['median_tok_s']:,.0f}" if name in bm else "-"})
    df = pd.DataFrame(rows)
    save_table(df, "table_part1", "Part I - mean [min-max] over seeds 42/43/44 (n-gram: one deterministic run). "
               "*n-gram throughput: plain-Python CPU lookups, not comparable with the neural workload.")
    return df


def part2_table():
    rows = []
    for name, tag in CHOSEN_PART2.items():
        rs = runs(tag)
        if not rs:
            continue
        gaps = [gap_at_best(f"{tag}{s}") for s in SEEDS if (RES / f"{tag}{s}_result.json").exists()]
        rows.append({"setting": name, "tag": tag, "trainable params": f"{rs[0]['params_trainable']:,}",
                     "val ppl": ms([r["best_val_ppl"] for r in rs]), "test ppl": ms([r["test_ppl"] for r in rs]),
                     "test ppl spread": f"{np.ptp([r['test_ppl'] for r in rs]):.1f}",
                     "best epoch": ms([r["best_epoch"] for r in rs], "{:.0f}"),
                     "val-train gap (nats)": ms(gaps, "{:.2f}"), "seeds": len(rs)})
    df = pd.DataFrame(rows)
    save_table(df, "table_part2", "Part II - same LSTM, learning rate and budget; mean [min-max] over training seeds. "
               "Spread = max-min test ppl across seeds (training stability).")
    c = RES / "sgns_cost.json"
    if c.exists():
        j = json.loads(c.read_text())
        print(f"\nOwn Skip-gram embeddings: {j['train_time_s']:.1f}s on {j['device']} ({j['epochs']} epochs, "
              f"{j['train_tokens']:,} tokens) - reported separately from LM training.")
    return df


# ---------------------------------------------------------------- figures
def curves_panel(ax, tag, title):
    f = RES / f"{tag}_log.csv"
    if not f.exists():
        ax.set_visible(False); return
    d = pd.read_csv(f)
    ax.plot(d.epoch, d.train_loss, color=BLUE, lw=2, label="train")
    ax.plot(d.epoch, d.val_loss, color=ORANGE, lw=2, label="validation")
    b = d.val_loss.idxmin()
    ax.plot([d.epoch[b]], [d.val_loss[b]], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2)
    ax.set_title(title, loc="left", fontsize=10, color=INK); ax.set_xlabel("epoch")


def fig_curves(items, fname):
    items = [(n, t) for n, t in items if (RES / f"{t}_log.csv").exists()]
    if not items:
        return
    cols = min(3, len(items)); rws = int(np.ceil(len(items) / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(4.6 * cols, 3.3 * rws), sharey=True, squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)
    for ax, (n, t) in zip(axs.ravel(), items):
        ax.set_visible(True); curves_panel(ax, t, n)
    for r in range(rws):
        axs[r, 0].set_ylabel("cross-entropy (nats/token)")
    h, l = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2)
    fig.text(0.01, 0.0, "seed 42; dot = best validation epoch; train loss measured in eval mode (no dropout)", color=MUTED, fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95)); plt.savefig(RES / fname, dpi=200); plt.close(fig)
    print("saved", RES / fname)


def fig_part3():
    f = RES / "classify_summary.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    models = [m for m in g.model.unique() if m != "TF-IDF (no LM)"]
    pools = [("last", BLUE, "last-token state"), ("mean", ORANGE, "mean over tokens"), ("embmean", AQUA, "mean input embedding (no LM layer)")]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for k, (p, col, lab) in enumerate(pools):
        d = g[g.pooling == p].set_index("model").reindex(models)
        x = np.arange(len(models)) + (k - 1) * 0.22
        ax.vlines(x, d.f1_min, d.f1_max, color=col, lw=2, alpha=0.9)
        ax.plot(x, d.f1_mean, "o", ms=9, color=col, mec=SURFACE, mew=2, label=lab)
    t = g[g.model == "TF-IDF (no LM)"]
    if len(t):
        ax.axhline(t.f1_mean.iloc[0], ls="--", lw=1.5, color=MUTED)
        ax.text(len(models) - 0.5, t.f1_mean.iloc[0] + 0.004, "TF-IDF, no language model", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("(", "\n(") for m in models], fontsize=8)
    ax.set_ylabel("test macro-F1: mean (dot), min-max (line)"); ax.legend(loc="lower right")
    ax.set_title("Part III - downstream ESG classification (dots, not bars: the axis does not start at 0)", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(RES / "fig_part3_f1.png", dpi=200); plt.close(fig)
    print("saved", RES / "fig_part3_f1.png")
    cols = ["model", "pooling", "lm_ppl", "val_f1", "test_acc", "f1_mean", "f1_env", "f1_min", "f1_max", "n_runs"]
    (RES / "table_part3.md").write_text("Part III - test macro-F1 over LM seeds x split seeds\n\n" + g[cols].round(3).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    part1_table(); part2_table()
    fig_curves(list(CHOSEN.items()), "fig_part1_curves.png")
    fig_curves([(n.replace("LSTM ", ""), t) for n, t in CHOSEN_PART2.items()], "fig_part2_curves.png")
    fig_part3()
