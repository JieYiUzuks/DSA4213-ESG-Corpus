"""
classify.py - Part III: do frozen language-model representations help ESG classification?
Dataset: ESGBERT/environmental_2k (2,000 sentences from company reports; label env=1 means
"environmental"). Only a train split exists, so we build our own STRATIFIED split:
60% train / 20% val / 20% test, repeated for SPLIT_SEEDS to see split-to-split variation.
Exact duplicate sentences are dropped BEFORE splitting (no train/test leakage).

Text is tokenized with the SAME tokenizer + vocab as Parts I-II, truncated to BLOCK_SIZE
tokens and padded at the END (lengths tracked, padding excluded from pooling).

Representations (LM frozen, eval mode, no dropout):
    last    = hidden state at the last real token (final state for RNN/LSTM; for the causal
              Transformer only the last token has seen the whole sentence)
    mean    = average of hidden states over real tokens
    embmean = average of the model's own INPUT embeddings (no sequence layer; a control)
    tfidf   = TF-IDF bag of words, no language model at all (lexical baseline)
Classifier (identical everywhere): [StandardScaler +] logistic regression with
class_weight="balanced"; C chosen on the validation set by macro-F1.
Each LM is evaluated for seeds 42/43/44 of its training run (checkpoints from Parts I-II).

Run from "Assignment 1":  python src/classify.py
Needs: pip install datasets scikit-learn pandas
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import prepare, tokenize_paragraph, BLOCK_SIZE, PAD, UNK, SEED
from generate import load_model, CHOSEN

RES = Path("results")
LABELS = ["not environmental", "environmental"]
SPLIT_SEEDS = [42, 43, 44]

MODELS = {                                    # display name -> base checkpoint tag (LM seed 42)
    "RNN": CHOSEN["RNN"],
    "LSTM": CHOSEN["LSTM"],
    "Transformer": CHOSEN["Transformer"],
    "LSTM+SGNS(frozen)": "lstm_sgns_frozen",
    "LSTM+GloVe(frozen)": "lstm_glove_frozen",
    "LSTM+SGNS(init,trainable)": "lstm_sgns_init",
    "LSTM+GloVe(init,trainable)": "lstm_glove_init",
}
LM_SEED_SUFFIXES = ["", "_s43", "_s44"]       # LM seed 42 has no suffix
POOLINGS = ["last", "mean", "embmean"]
ERR_CONFIGS = [("Transformer", "mean"), ("LSTM", "mean")]   # error-analysis files


def load_esg():
    from datasets import load_dataset
    df = load_dataset("ESGBERT/environmental_2k")["train"].to_pandas()
    n0 = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates("text").reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after dropping exact duplicate sentences")
    y = df["env"].to_numpy().astype(int)
    print(f"label counts: {dict(zip(LABELS, np.bincount(y, minlength=2)))}  "
          f"majority-class accuracy {max(np.bincount(y)) / len(y):.1%}")
    return df["text"].tolist(), y


def make_split(y, seed):
    idx = np.arange(len(y))
    trval, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(trval, test_size=0.25, stratify=y[trval], random_state=seed)
    return tr, va, te


def encode_all(texts, stoi):
    X = torch.full((len(texts), BLOCK_SIZE), PAD, dtype=torch.long)
    lens, unk, tot, trunc = [], 0, 0, 0
    for i, t in enumerate(texts):
        raw = [stoi.get(w, UNK) for w in tokenize_paragraph(t)]
        trunc += len(raw) > BLOCK_SIZE
        ids = raw[:BLOCK_SIZE] or [UNK]
        X[i, :len(ids)] = torch.tensor(ids)
        lens.append(len(ids)); unk += sum(j == UNK for j in ids); tot += len(ids)
    return X, torch.tensor(lens), unk / tot, trunc / len(texts)


@torch.no_grad()
def features(model, X, lens, pooling, bs=256):
    out = []
    for i in range(0, len(X), bs):
        x, l = X[i:i + bs], lens[i:i + bs]
        h = model.emb(x) if pooling == "embmean" else model.hidden_states(x)
        if pooling == "last":
            f = h[torch.arange(len(x)), l - 1]
        else:                                               # mean over real tokens only
            m = (torch.arange(x.size(1))[None, :] < l[:, None]).unsqueeze(-1).float()
            f = (h * m).sum(1) / l.unsqueeze(-1).float()
        out.append(f)
    return torch.cat(out).numpy()


def fit_eval(Ftr, ytr, Fva, yva, Fte, yte, scale=True, Cs=(0.001, 0.01, 0.1, 1.0)):
    if scale:
        sc = StandardScaler().fit(Ftr)
        Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ftr, ytr)
        f1 = f1_score(yva, clf.predict(Fva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    f1v, C, clf = best
    pred = clf.predict(Fte)
    return {"C": C, "val_f1": f1v, "test_acc": accuracy_score(yte, pred),
            "test_f1": f1_score(yte, pred, average="macro"),
            "test_f1_env": f1_score(yte, pred, pos_label=1)}, pred


def main():
    data, itos = prepare()
    V = len(itos); stoi = {w: i for i, w in enumerate(itos)}
    texts, y = load_esg()
    X, lens, unk_rate, trunc_rate = encode_all(texts, stoi)
    print(f"<unk> rate on this dataset: {unk_rate:.1%}  (Part I test set: ~4-5%)")
    print(f"mean length {lens.float().mean():.1f} tokens; {trunc_rate:.1%} of sentences "
          f"truncated at {BLOCK_SIZE} tokens")
    splits = {s: make_split(y, s) for s in SPLIT_SEEDS}
    print({k: len(v) for k, v in zip(("train", "val", "test"), splits[SPLIT_SEEDS[0]])})

    rows, preds = [], {}

    def record(name, tag, lm_seed, pool, ppl, ssd, r, pred, te):
        rows.append({"model": name, "tag": tag, "lm_seed": lm_seed, "split_seed": ssd,
                     "pooling": pool, "lm_test_ppl": ppl, **r})
        if lm_seed == "_s42" and ssd == SPLIT_SEEDS[0]:
            preds[(name, pool)] = (pred, te)

    # lexical baseline: TF-IDF, vocabulary fit on the training split only
    for ssd, (tr, va, te) in splits.items():
        vec = TfidfVectorizer(min_df=2, sublinear_tf=True).fit([texts[i] for i in tr])
        T = lambda idx: vec.transform([texts[i] for i in idx])
        r, pred = fit_eval(T(tr), y[tr], T(va), y[va], T(te), y[te], scale=False,
                           Cs=(0.1, 1.0, 10.0, 100.0))
        record("TF-IDF (no LM)", "-", "-", "bow", float("nan"), ssd, r, pred, te)

    for name, base in MODELS.items():
        for sfx in LM_SEED_SUFFIXES:
            tag = base + sfx
            if not (RES / f"{tag}.pt").exists():
                print("missing checkpoint, skipping:", tag); continue
            model = load_model(tag, V, "cpu")
            ppl = json.loads((RES / f"{tag}_result.json").read_text())["test_ppl"]
            for pool in POOLINGS:
                F = features(model, X, lens, pool)
                for ssd, (tr, va, te) in splits.items():
                    r, pred = fit_eval(F[tr], y[tr], F[va], y[va], F[te], y[te])
                    record(name, tag, sfx or "_s42", pool, ppl, ssd, r, pred, te)
            print("done", tag)

    df = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "classify_all_runs.csv", index=False)
    g = df.groupby(["model", "pooling"]).agg(
        lm_ppl=("lm_test_ppl", "mean"), val_f1=("val_f1", "mean"),
        test_acc=("test_acc", "mean"), f1_mean=("test_f1", "mean"),
        f1_env=("test_f1_env", "mean"), f1_min=("test_f1", "min"),
        f1_max=("test_f1", "max"), n_runs=("test_f1", "count")).round(4)
    g.to_csv(RES / "classify_summary.csv")
    print("\n", g.to_string())

    for name, pool in ERR_CONFIGS:      # error analysis: LM seed 42, first split
        if (name, pool) not in preds: continue
        pred, te = preds[(name, pool)]
        yt = y[te]
        wrong = np.where(pred != yt)[0]
        fp = wrong[pred[wrong] == 1]; fn = wrong[pred[wrong] == 0]
        rng = np.random.default_rng(SEED)
        take = lambda a: rng.choice(a, size=min(5, len(a)), replace=False)
        lines = [f"# Errors: {name}, {pool} pooling - {len(wrong)} errors of {len(te)} "
                 f"({len(fp)} false positives, {len(fn)} false negatives)\n",
                 "## False positives (true: not environmental, predicted: environmental)"]
        lines += [f"- {texts[te[p]][:300]}" for p in take(fp)] if len(fp) else ["- none"]
        lines += ["\n## False negatives (true: environmental, predicted: not environmental)"]
        lines += [f"- {texts[te[p]][:300]}" for p in take(fn)] if len(fn) else ["- none"]
        fn_name = RES / f"errors_{name.replace('+', '_').replace('(', '').replace(')', '')}_{pool}.md"
        fn_name.write_text("\n".join(lines))
        print("wrote", fn_name)


if __name__ == "__main__":
    main()
