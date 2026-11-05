"""Okapi BM25 over a sparse term-frequency matrix.

Scoring (Lucene-style idf, non-negative):

    idf(t)      = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))
    w(t, d)     = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    score(q, d) = sum over query terms t of q_t * w(t, d)

Weights are materialised once per ``(k1, b)`` so a parameter sweep only recomputes the
weight matrix, not the counts.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import scipy.sparse as sp


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_ids: np.ndarray | None = None
        self.vocab: dict[str, int] = {}
        self.tf: sp.csr_matrix | None = None
        self.doc_len: np.ndarray | None = None
        self.idf: np.ndarray | None = None
        self.weights: sp.csc_matrix | None = None

    # ------------------------------------------------------------------ building
    def fit(self, doc_ids: list[int], docs_tokens: list[list[str]]) -> BM25Index:
        if len(doc_ids) != len(docs_tokens) or not doc_ids:
            raise ValueError("doc_ids and docs_tokens must be non-empty and the same length")
        rows, cols, vals = [], [], []
        vocab: dict[str, int] = {}
        for i, toks in enumerate(docs_tokens):
            for term, count in Counter(toks).items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                vals.append(count)
        n = len(doc_ids)
        self.vocab = vocab
        self.doc_ids = np.asarray(doc_ids, dtype=np.int64)
        self.tf = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                                shape=(n, len(vocab)))
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel()
        df = np.diff(self.tf.tocsc().indptr).astype(np.float64)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._reweight()
        return self

    def set_params(self, k1: float, b: float) -> BM25Index:
        self.k1, self.b = k1, b
        self._reweight()
        return self

    def _reweight(self) -> None:
        coo = self.tf.tocoo()
        avgdl = max(float(self.doc_len.mean()), 1e-9)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[coo.row] / avgdl)
        w = self.idf[coo.col] * coo.data * (self.k1 + 1.0) / (coo.data + norm)
        self.weights = sp.csc_matrix((w.astype(np.float32), (coo.row, coo.col)),
                                     shape=self.tf.shape)

    # ------------------------------------------------------------------ scoring
    def _query_vector(self, query: Iterable[str] | Mapping[str, float]):
        weights = dict(query) if isinstance(query, Mapping) else Counter(query)
        idx, w = [], []
        for term, weight in weights.items():
            j = self.vocab.get(term)
            if j is not None:
                idx.append(j)
                w.append(weight)
        return np.asarray(idx, dtype=np.int64), np.asarray(w, dtype=np.float32)

    def score(self, query: Iterable[str] | Mapping[str, float]) -> np.ndarray:
        """BM25 score of every document (corpus order); unknown terms are ignored."""
        idx, w = self._query_vector(query)
        if idx.size == 0:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, idx] @ w).ravel()

    def search(self, query, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        s = self.score(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return self.doc_ids[top], s[top]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "tf.npz", self.tf)
        np.save(d / "doc_ids.npy", self.doc_ids)
        (d / "meta.json").write_text(json.dumps({"k1": self.k1, "b": self.b,
                                                 "vocab": self.vocab}))

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        idx = cls(meta["k1"], meta["b"])
        idx.vocab = meta["vocab"]
        idx.tf = sp.load_npz(d / "tf.npz").tocsr()
        idx.doc_ids = np.load(d / "doc_ids.npy")
        n = idx.tf.shape[0]
        idx.doc_len = np.asarray(idx.tf.sum(axis=1)).ravel()
        df = np.diff(idx.tf.tocsc().indptr).astype(np.float64)
        idx.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        idx._reweight()
        return idx
