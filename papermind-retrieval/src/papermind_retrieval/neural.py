"""A small trainable dual-encoder over hashed unigram and bigram features.

This is deliberately a compact model that trains on CPU in a minute: a shared
``EmbeddingBag`` tower with a projection, trained with an in-batch InfoNCE loss. It is not
a substitute for a pretrained transformer encoder; it exists to test whether a lightly
trained neural signal adds anything to lexical retrieval on a small benchmark.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import torch
from papermind_ingest import tokenize
from torch import nn


def hashed_ids(text: str, buckets: int) -> list[int]:
    toks = tokenize(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
    return [zlib.crc32(g.encode()) % buckets for g in grams] or [0]


class HashedEncoder(nn.Module):
    def __init__(self, buckets: int = 1 << 16, dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        self.buckets = buckets
        with torch.random.fork_rng():  # seeded initialisation without touching global state
            torch.manual_seed(seed)
            self.bag = nn.EmbeddingBag(buckets, dim, mode="mean")
            self.proj = nn.Linear(dim, dim)
            nn.init.normal_(self.bag.weight, std=0.1)

    def forward(self, ids: list[list[int]]) -> torch.Tensor:
        flat = torch.tensor([i for seq in ids for i in seq], dtype=torch.long)
        offsets = torch.tensor(np.cumsum([0] + [len(s) for s in ids[:-1]]), dtype=torch.long)
        return nn.functional.normalize(self.proj(self.bag(flat, offsets)), dim=-1)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch: int = 512) -> np.ndarray:
        self.eval()
        out = [self([hashed_ids(t, self.buckets) for t in texts[i:i + batch]]).numpy()
               for i in range(0, len(texts), batch)]
        return np.concatenate(out)


def train_pairs(model: HashedEncoder, pairs: Sequence[tuple[str, str]], epochs=10, batch=64,
                lr=3e-3, temperature=0.05, seed=0) -> list[float]:
    """Contrastive training on (query, positive) pairs with in-batch negatives."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    q_ids = [hashed_ids(q, model.buckets) for q, _ in pairs]
    d_ids = [hashed_ids(d, model.buckets) for _, d in pairs]
    losses = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(pairs))
        total, nb = 0.0, 0
        for i in range(0, len(order) - 1, batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            q = model([q_ids[j] for j in idx])
            d = model([d_ids[j] for j in idx])
            logits = q @ d.T / temperature
            labels = torch.arange(len(idx))
            loss = (nn.functional.cross_entropy(logits, labels)
                    + nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        losses.append(total / max(nb, 1))
    return losses


class NeuralSystem:
    def __init__(self, model: HashedEncoder, doc_texts: list[str]):
        self.model = model
        self.emb = model.encode(doc_texts)

    def scores(self, text: str) -> np.ndarray:
        return (self.emb @ self.model.encode([text]).T).ravel()
