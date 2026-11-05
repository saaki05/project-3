"""Ranking metrics with binary relevance, and paired significance tests."""

from __future__ import annotations

import math

import numpy as np
from papermind_ingest import Claim


def rank_docs(scores: np.ndarray, doc_ids: list[int], k: int) -> list[int]:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [doc_ids[i] for i in top]


def ndcg_at_k(ranked: list[int], relevant: frozenset[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(ranked: list[int], relevant: frozenset[int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[int], relevant: frozenset[int], k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def per_query_metrics(run: dict[int, list[int]], claims: list[Claim]) -> dict[str, np.ndarray]:
    """Per-claim metrics over claims that have at least one relevant document."""
    rows = {"ndcg@10": [], "mrr@10": [], "recall@10": [], "recall@100": []}
    for c in claims:
        if not c.has_evidence:
            continue
        r, rel = run[c.claim_id], c.relevant_docs
        rows["ndcg@10"].append(ndcg_at_k(r, rel, 10))
        rows["mrr@10"].append(mrr_at_k(r, rel, 10))
        rows["recall@10"].append(recall_at_k(r, rel, 10))
        rows["recall@100"].append(recall_at_k(r, rel, 100))
    return {k: np.asarray(v) for k, v in rows.items()}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a) - mean(b) over queries, with a two-sided p-value."""
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference."""
    d = a - b
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float(((perm >= abs(d.mean()) - 1e-12).sum() + 1) / (n_perm + 1))
