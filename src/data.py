"""
data.py - corpus cleaning, tokenization, splits, vocab, batching.
Run from the "Assignment 1" folder:  python src/data.py

SPLIT DESIGN (v3): every report is cut into consecutive blocks of SPLIT_BLOCK = 1024 tokens
(a multiple of the 64-token window). Within a report, block i goes to VAL if i % 10 == (4+r) % 10
and to TEST if i % 10 == (9+r) % 10, where r is the report index; all other blocks are TRAIN.
So val/test are ~10% each, spread over every section of every report (not just the tail, which
is mostly appendices / assurance statements), with a different phase per report. Windows never
cross a block boundary. The trailing <1024 tokens of each report are discarded (~1% of tokens).
Vocabulary is built from TRAIN only. The split is deterministic (no randomness).
"""
import re, json, random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

SEED = 42
CORPUS_PATH = Path("data/sustainability_reports_corpus.txt")
PROC_DIR = Path("data/processed")
BLOCK_SIZE = 64          # context length (same for all neural models)
SPLIT_BLOCK = 1024       # split granularity in tokens (must be a multiple of BLOCK_SIZE)
SPLIT_ID = "blocks1024_rot10_v3"   # cache key: a stale cache from another split design is rebuilt
MIN_FREQ = 3             # word must appear >= 3 times in TRAIN to be in vocab
MAX_VOCAB = 10000        # cap incl. special tokens
SPECIALS = ["<pad>", "<unk>", "<eos>"]   # ids 0, 1, 2
PAD, UNK, EOS = 0, 1, 2

MARKER_RE = re.compile(r"<<<REPORT_START[^>]*>>>")
# Protected identifiers (kept as single tokens, NOT collapsed into <num>):
#   "scope 1/2/3" -> scope_1/scope_2/scope_3 ; years 2010-2069 -> year_2030 etc.
PROTECT = [
    (re.compile(r"\bscope\s*([123])\b"), r" scope_\1 "),
    (re.compile(r"\b(20[1-6]\d)\b"), r" year_\1 "),
]
# standalone numbers only (not digits inside words such as co2, tco2e, s2); does not eat
# a trailing comma / full stop
NUM_RE = re.compile(r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")
TOKEN_RE = re.compile(r"<num>|[a-z][a-z0-9_]*(?:['\u2019-][a-z0-9_]+)*|[.,;:!?()%]")


# ---------- cleaning / tokenization ----------
def split_reports(text):
    """Split the raw file into one string per report (using the markers)."""
    parts = MARKER_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_paragraph(par):
    par = par.lower()
    par = par.replace("\u2019", "'")
    par = re.sub(r"[\u2022\u2013\u2014\u25cf\u25aa]", " ", par)     # bullets, dashes
    for pat, rep in PROTECT:                                        # scope_1, year_2030, ...
        par = pat.sub(rep, par)
    par = NUM_RE.sub(" <num> ", par)                                # other numbers -> <num>
    return TOKEN_RE.findall(par)


def tokenize_report(report):
    """Paragraph (blank-line separated) -> tokens ending with <eos>."""
    tokens = []
    for par in re.split(r"\n\s*\n", report):
        toks = tokenize_paragraph(par)
        if len(toks) >= 3:                 # drop tiny fragments (page numbers etc.)
            tokens.extend(toks + ["<eos>"])
    return tokens


# ---------- splits: rotating blocks (see module docstring) ----------
def block_split(reports_tokens, block=SPLIT_BLOCK):
    assert block % BLOCK_SIZE == 0
    train, val, test = [], [], []
    info = {"dropped_tokens": 0, "blocks": {"train": 0, "val": 0, "test": 0}, "per_report": []}
    for r, toks in enumerate(reports_tokens):
        nb = len(toks) // block
        info["dropped_tokens"] += len(toks) - nb * block
        cnt = {"train": 0, "val": 0, "test": 0}
        for i in range(nb):
            seg = toks[i * block:(i + 1) * block]
            phase = i % 10
            if phase == (4 + r) % 10:
                val += seg; cnt["val"] += 1
            elif phase == (9 + r) % 10:
                test += seg; cnt["test"] += 1
            else:
                train += seg; cnt["train"] += 1
        for k, v in cnt.items():
            info["blocks"][k] += v
        info["per_report"].append(cnt)
    return train, val, test, info


# ---------- vocab (TRAIN ONLY) ----------
def build_vocab(train_tokens):
    counts = Counter(t for t in train_tokens if t not in SPECIALS)
    words = [w for w, c in counts.most_common() if c >= MIN_FREQ]
    words = words[: MAX_VOCAB - len(SPECIALS)]
    itos = SPECIALS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(tokens, stoi):
    return [stoi.get(t, UNK) for t in tokens]


# ---------- windows / dataloaders ----------
class WindowDataset(Dataset):
    """Non-overlapping windows of BLOCK_SIZE+1 ids -> (input, target) shifted by one.
    `ids` is a concatenation of whole SPLIT_BLOCK-token blocks; windows are built inside each
    block, so no window (and no target) ever crosses a block boundary. A window that runs past
    the end of its block is padded with <pad>; the loss must use ignore_index=PAD."""
    def __init__(self, ids, block_size=BLOCK_SIZE, segment=SPLIT_BLOCK):
        self.windows = []
        step = block_size
        for b0 in range(0, len(ids), segment):
            seg = ids[b0 : b0 + segment]
            for s in range(0, len(seg) - 1, step):
                w = seg[s : s + block_size + 1]
                if len(w) < 2:
                    continue
                w = w + [PAD] * (block_size + 1 - len(w))
                self.windows.append(w)
        self.windows = torch.tensor(self.windows, dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        return w[:-1], w[1:]


def prepare(force=False):
    """Run the full pipeline once and cache to data/processed/."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROC_DIR / "ids.pt"
    if cache.exists() and not force:
        cached = torch.load(cache)
        if cached.get("split_id") == SPLIT_ID:
            return cached, json.loads((PROC_DIR / "vocab.json").read_text())
        print("[data] cache was built with a different split design -> rebuilding")

    text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
    reports = split_reports(text)
    reports_tokens = [tokenize_report(r) for r in reports]
    train_t, val_t, test_t, info = block_split(reports_tokens)
    itos, stoi = build_vocab(train_t)

    data = {
        "train": encode(train_t, stoi),
        "val": encode(val_t, stoi),
        "test": encode(test_t, stoi),
        "n_reports": len(reports),
        "split_id": SPLIT_ID,
        "split_info": info,
        "report_lengths": [len(t) for t in reports_tokens],
    }
    torch.save(data, cache)   # token ids only; vocabulary is stored separately
    (PROC_DIR / "vocab.json").write_text(json.dumps(itos))
    return data, itos


def get_loaders(batch_size=64, block_size=BLOCK_SIZE, seed=SEED):
    data, itos = prepare()
    g = torch.Generator().manual_seed(seed)   # controls training-batch order
    train = DataLoader(WindowDataset(data["train"], block_size), batch_size, shuffle=True, generator=g)
    val = DataLoader(WindowDataset(data["val"], block_size), batch_size)
    test = DataLoader(WindowDataset(data["test"], block_size), batch_size)
    return train, val, test, itos


# ---------- stats for the report ----------
if __name__ == "__main__":
    random.seed(SEED); torch.manual_seed(SEED)
    data, itos = prepare(force=True)
    print(f"Reports found      : {data['n_reports']}")
    for k in ("train", "val", "test"):
        ids = data[k]
        unk = sum(1 for i in ids if i == UNK) / max(1, len(ids))
        print(f"{k:5s} tokens: {len(ids):>9,d}   <unk> rate: {unk:.2%}")
    print(f"Vocab size         : {len(itos):,d} (min_freq={MIN_FREQ}, cap={MAX_VOCAB})")
    inf = data["split_info"]
    print(f"Blocks (x{SPLIT_BLOCK} tokens): {inf['blocks']}   discarded tail tokens: {inf['dropped_tokens']:,d} "
          f"of {sum(data['report_lengths']):,d}")
    print("Blocks per report  :", [(c['train'], c['val'], c['test']) for c in inf['per_report']], "(train, val, test)")
    # overlap diagnostic: share of val/test 6-grams that also occur in train (boilerplate / adjacency)
    six = lambda x: list(zip(*[x[i:] for i in range(6)]))
    seen = set(six(data["train"]))
    for k in ("val", "test"):
        g = six(data[k])
        print(f"{k:5s} 6-grams also in train: {sum(q in seen for q in g) / len(g):.1%}")
    prot = [w for w in itos if w.startswith(("scope_", "year_"))]
    print("Protected in vocab :", len(prot), prot)
    print("Most common        :", itos[3:23])
    print("Sample (train)     :", " ".join(itos[i] for i in data["train"][1000:1060]))
