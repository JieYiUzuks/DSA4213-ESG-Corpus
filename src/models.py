"""
models.py - RNN / LSTM / causal Transformer language models.
All models: input ids (B,T) -> logits (B,T,V). Input and output embeddings are UNTIED
(needed so the input table can be frozen in Part II).
Padding is only ever at the END of a window, and every model is causal, so real tokens
never see pad positions (no key-padding mask needed). Pad targets are ignored in the loss.
"""
import torch
import torch.nn as nn


class BaseLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx=0, emb_weights=None, freeze_emb=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if emb_weights is not None:                       # Part II: pre-built table
            self.emb.weight.data.copy_(torch.as_tensor(emb_weights, dtype=torch.float))
        if freeze_emb:                                    # whole table fixed
            self.emb.weight.requires_grad = False

    def forward(self, x):
        return self.out(self.drop(self.hidden_states(x)))


class RecurrentLM(BaseLM):
    def __init__(self, vocab_size, kind="lstm", emb_dim=100, hidden=256, layers=1,
                 dropout=0.3, **kw):
        super().__init__(vocab_size, emb_dim, **kw)
        cls = {"lstm": nn.LSTM, "rnn": nn.RNN}[kind]
        self.rnn = cls(emb_dim, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def hidden_states(self, x):                           # (B,T,H); also used in Part III
        h, _ = self.rnn(self.drop(self.emb(x)))
        return h


class TransformerLM(BaseLM):
    def __init__(self, vocab_size, emb_dim=100, layers=2, heads=4, ff=256, dropout=0.3,
                 block_size=64, **kw):
        super().__init__(vocab_size, emb_dim, **kw)
        self.pos = nn.Embedding(block_size, emb_dim)      # learned positions
        layer = nn.TransformerEncoderLayer(emb_dim, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(emb_dim),
                                         enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, vocab_size)

    def hidden_states(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.emb(x) + self.pos(pos))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.enc(h, mask=mask, is_causal=True)     # causal self-attention


def build_model(name, vocab_size, emb_dim=100, hidden=256, layers=None, dropout=0.3,
                heads=4, ff=256, block_size=64, **kw):
    if name in ("rnn", "lstm"):
        return RecurrentLM(vocab_size, kind=name, emb_dim=emb_dim, hidden=hidden,
                           layers=layers or 1, dropout=dropout, **kw)
    if name == "transformer":
        return TransformerLM(vocab_size, emb_dim=emb_dim, layers=layers or 2, heads=heads,
                             ff=ff, dropout=dropout, block_size=block_size, **kw)
    raise ValueError(name)