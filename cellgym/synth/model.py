"""Small decoder-only transformer for the plan language (pre-norm, learned positions).

    model = GPT(Config())                 # ~32M parameters at the defaults
    logits = model(idx)                   # (B, T, vocab)
    ids = sample(model, prompt_ids, n=4)  # n lists of generated ids, cut at and including EOS
    save(model, cfg, "best.pt"); model = load("best.pt")
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cellgym.synth import tokens as T


@dataclass
class Config:
    vocab: int = len(T.VOCAB)
    d_model: int = 512
    layers: int = 10
    heads: int = 8
    max_len: int = 512
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.out = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cache=None):
        """cache: None (full causal pass) or a [k, v] pair of (B, H, S, D) tensors, extended in place."""
        B, T_, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T_, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if cache is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if cache:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            cache[:] = [k, v]
            # prefill (T_ > 1, empty cache) is causal; a single new token attends to everything
            a = F.scaled_dot_product_attention(q, k, v, is_causal=T_ > 1)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T_, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, c: Config | None = None):
        super().__init__()
        self.cfg = c = c or Config()
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
        self.apply(self._init)
        for b in self.blocks:                                    # scale residual projections
            for lin in (b.proj, b.out):
                nn.init.normal_(lin.weight, 0.0, 0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, caches=None, start: int = 0):
        """idx (B, T) -> logits (B, T, vocab). caches: per-layer lists for incremental decoding."""
        T_ = idx.shape[1]
        if start + T_ > self.cfg.max_len:
            raise ValueError(f"sequence length {start + T_} exceeds max_len {self.cfg.max_len}")
        x = self.tok(idx) + self.pos(torch.arange(start, start + T_, device=idx.device))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i])
        return self.head(self.ln(x))


# -- checkpoints ---------------------------------------------------------------------------

def save(model: GPT, cfg: Config, path) -> None:
    torch.save({"state": model.state_dict(), "config": asdict(cfg)}, path)


def load(path, device="cpu") -> GPT:
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


# -- sampling ------------------------------------------------------------------------------

@torch.no_grad()
def sample(model: GPT, prompt_ids: list[int], n: int = 1, temperature: float = 1.0,
           max_new: int = 320, top_k: int | None = None) -> list[list[int]]:
    """n continuations of the prompt, each cut at and including EOS (or max_new tokens).

    temperature 0 (or top_k 1) is greedy. Uses a KV cache; the n samples run as one batch.
    """
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    max_new = min(max_new, model.cfg.max_len - len(prompt_ids))
    caches = [[] for _ in model.blocks]
    x = torch.tensor([prompt_ids] * n, device=dev)
    out = torch.zeros(n, 0, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    pos = 0
    for _ in range(max_new):
        logits = model(x, caches, start=pos)[:, -1].float()
        pos += x.shape[1]
        if top_k:
            kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if temperature <= 0 or top_k == 1:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(1)
        nxt = nxt.masked_fill(done, T.PAD)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == T.EOS
        if done.all():
            break
        x = nxt[:, None]
    if was_training:
        model.train()
    res = []
    for row in out.tolist():
        res.append(row[:row.index(T.EOS) + 1] if T.EOS in row else row)
    return res
