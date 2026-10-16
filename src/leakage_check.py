"""
leakage_check.py - does the Part III dataset (ESGBERT/environmental_2k) overlap with the text the
language models were trained on?

Run from "Assignment 1" (needs data/sustainability_reports_corpus.txt and internet for the dataset):
    python src/leakage_check.py
    python src/leakage_check.py --sentences my_sentences.txt     # offline test: one sentence per line

What it does
  * Tokenises each ESGBERT sentence with the SAME tokenizer as the corpus (src/data.py), so numbers
    become <num>, scope/year tokens are protected, and so on: it compares what the models could have seen.
  * Compares against (a) the TRAIN split of the reports (the only text the language models were trained on;
    validation text was used for early stopping only) and (b) all nine reports.
  * Reports, over sentences with >= MIN_TOKENS tokens:
        exact   : the whole sentence appears as one contiguous token sequence
        cover50 : at least half of its 8-grams occur in the text
        any8    : at least one 8-gram occurs in the text
  * Reference: the same 8-gram overlap between the corpus' own validation/test blocks and its train
    blocks (how much overlap two parts of the SAME report collection show).
Writes results/leakage_check.json and results/leakage_examples.md (flagged sentences with the matching text).
"""
import argparse
import json
from pathlib import Path

from data import (CORPUS_PATH, split_reports, tokenize_report, tokenize_paragraph, block_split)

RES = Path("results")
N = 8                 # n-gram length for the overlap measure
MIN_TOKENS = 8        # shorter sentences are not meaningful to match (and are excluded from the stats)


def ngrams(toks, n=N):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def pad(toks):
    return " " + " ".join(toks) + " "


def stats(sent_toks, sent_idx, corpus_tokens, corpus_str, label):
    grams = ngrams(corpus_tokens)
    exact, cover50, any8, rows = 0, 0, 0, []
    for i in sent_idx:
        t = sent_toks[i]
        g = ngrams(t)
        hit = sum(x in grams for x in g) / max(len(g), 1)
        e = pad(t) in corpus_str
        exact += e; cover50 += hit >= 0.5; any8 += hit > 0
        if e or hit >= 0.5:
            rows.append((i, e, hit))
    n = len(sent_idx)
    out = {"corpus": label, "sentences_checked": n,
           "exact": exact, "exact_pct": 100 * exact / n,
           "cover50": cover50, "cover50_pct": 100 * cover50 / n,
           "any8": any8, "any8_pct": 100 * any8 / n}
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", help="text file, one sentence per line (offline test); default: ESGBERT")
    a = ap.parse_args()

    # ---- corpus tokens: train blocks vs all
    reports = split_reports(CORPUS_PATH.read_text(encoding="utf-8", errors="ignore"))
    rep_tokens = [tokenize_report(r) for r in reports]
    train_t, val_t, test_t, _ = block_split(rep_tokens)
    all_t = [t for r in rep_tokens for t in r]
    train_s, all_s = pad(train_t), pad(all_t)

    # ---- sentences
    if a.sentences:
        texts = [l.strip() for l in Path(a.sentences).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        from classify import load_esg
        texts, _ = load_esg()
    sent_toks = [tokenize_paragraph(t) for t in texts]
    keep = [i for i, t in enumerate(sent_toks) if len(t) >= MIN_TOKENS]
    print(f"\nESGBERT sentences: {len(texts)}; with >= {MIN_TOKENS} tokens (checked): {len(keep)}")

    res_train, rows_train = stats(sent_toks, keep, train_t, train_s, "train blocks (what the LMs trained on)")
    res_all, rows_all = stats(sent_toks, keep, all_t, all_s, "all nine reports")

    # ---- reference: corpus val/test vs train (8-grams)
    tr_g = ngrams(train_t)
    ref = {}
    for name, toks in [("val", val_t), ("test", test_t)]:
        g = ngrams(toks)
        ref[name] = 100 * sum(x in tr_g for x in g) / max(len(g), 1)
    print(f"Reference: {N}-grams of the corpus' own val/test blocks that also occur in train: "
          f"val {ref['val']:.1f}%, test {ref['test']:.1f}%")

    for r in (res_train, res_all):
        print(f"\n{r['corpus']}\n  exact whole-sentence match : {r['exact']:4d} ({r['exact_pct']:.1f}%)"
              f"\n  >=50% of 8-grams matched   : {r['cover50']:4d} ({r['cover50_pct']:.1f}%)"
              f"\n  any 8-gram matched          : {r['any8']:4d} ({r['any8_pct']:.1f}%)")

    # ---- examples
    flagged = sorted({i for i, _, _ in rows_train} | {i for i, _, _ in rows_all})
    RES.mkdir(exist_ok=True)
    lines = ["# Sentences flagged (exact match or >=50% of 8-grams in the reports)\n",
             f"{len(flagged)} flagged of {len(keep)} checked.\n"]
    for i in flagged[:40]:
        in_train = any(i == j for j, _, _ in rows_train)
        lines.append(f"- [{'train' if in_train else 'reports only (not train)'}] {texts[i]}")
    (RES / "leakage_examples.md").write_text("\n".join(lines) + "\n")
    out = {"n_sentences": len(texts), "n_checked": len(keep), "min_tokens": MIN_TOKENS, "ngram": N,
           "vs_train": res_train, "vs_all_reports": res_all, "reference_corpus_overlap_pct": ref,
           "flagged_sentence_indices": flagged, "flagged_texts": [texts[i] for i in flagged]}
    (RES / "leakage_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/leakage_check.json and results/leakage_examples.md ({len(flagged)} flagged)")


if __name__ == "__main__":
    main()
