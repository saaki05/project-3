"""Retrieval systems with a common interface: ``scores(query_text) -> (N,) array``.

All systems score the same corpus in the same order, which lets fusion operate on score or
rank vectors directly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from papermind_ingest import Document, tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from papermind_retrieval.bm25 import BM25Index


def doc_text(doc: Document, field: str) -> str:
    if field == "title":
        return doc.title
    if field == "abstract":
        return doc.abstract
    if field == "title+abstract":
        return doc.text
    raise ValueError(f"unknown field {field!r}")


class BM25System:
    def __init__(self, corpus: dict[int, Document], k1=1.2, b=0.75, field="title+abstract",
                 stem=True, stop=True):
        self.stem, self.stop = stem, stop
        self.doc_ids = sorted(corpus)
        toks = [tokenize(doc_text(corpus[i], field), stem, stop) for i in self.doc_ids]
        self.index = BM25Index(k1, b).fit(self.doc_ids, toks)

    def tokens(self, text: str) -> list[str]:
        return tokenize(text, self.stem, self.stop)

    def set_params(self, k1: float, b: float):
        self.index.set_params(k1, b)
        return self

    def scores(self, text: str) -> np.ndarray:
        return self.index.score(self.tokens(text))


class RM3System:
    """BM25 with RM3-style pseudo-relevance feedback.

    The top ``fb_docs`` documents of an initial BM25 run contribute their terms, weighted by
    document score and in-document term frequency; the ``fb_terms`` heaviest new terms are
    interpolated with the original query using weight ``lam`` on the original.
    """

    def __init__(self, base: BM25System, fb_docs=5, fb_terms=10, lam=0.6):
        self.base, self.fb_docs, self.fb_terms, self.lam = base, fb_docs, fb_terms, lam

    def scores(self, text: str) -> np.ndarray:
        idx = self.base.index
        q = self.base.tokens(text)
        first = idx.score(q)
        if not q or first.max() <= 0:
            return first
        top = np.argsort(-first, kind="stable")[: self.fb_docs]
        w = first[top] / first[top].sum()
        inv_vocab = getattr(self, "_inv", None)
        if inv_vocab is None:
            inv_vocab = self._inv = {j: t for t, j in idx.vocab.items()}
        weight = Counter()
        for dw, d in zip(w, top, strict=True):
            row = idx.tf.getrow(int(d))
            tf = row.data / max(float(idx.doc_len[d]), 1.0)
            for j, f in zip(row.indices, tf, strict=True):
                weight[int(j)] += dw * f * idx.idf[j]
        expansion = {inv_vocab[j]: v for j, v in weight.most_common(self.fb_terms)}
        total = sum(expansion.values()) or 1.0
        qtf = Counter(q)
        qn = sum(qtf.values())
        mixed = {t: self.lam * c / qn for t, c in qtf.items()}
        for t, v in expansion.items():
            mixed[t] = mixed.get(t, 0.0) + (1 - self.lam) * v / total
        return idx.score(mixed)


class LSASystem:
    """TF-IDF followed by truncated SVD; cosine similarity in the latent space."""

    def __init__(self, corpus: dict[int, Document], dims=256, field="title+abstract", seed=0):
        self.doc_ids = sorted(corpus)
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=2)
        x = self.vec.fit_transform(doc_text(corpus[i], field) for i in self.doc_ids)
        dims = min(dims, x.shape[1] - 1, x.shape[0] - 1)
        self.svd = TruncatedSVD(dims, random_state=seed).fit(x)
        self.emb = self._normalise(self.svd.transform(x))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def scores(self, text: str) -> np.ndarray:
        q = self._normalise(self.svd.transform(self.vec.transform([text])))
        return (self.emb @ q.T).ravel()


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


class FusionSystem:
    """Combine systems by reciprocal rank fusion or weighted min-max score fusion."""

    def __init__(self, systems, method="rrf", weights=None, rrf_k=60):
        if method not in ("rrf", "minmax"):
            raise ValueError("method must be 'rrf' or 'minmax'")
        self.systems, self.method, self.rrf_k = systems, method, rrf_k
        self.weights = weights or [1.0] * len(systems)

    def scores(self, text: str) -> np.ndarray:
        parts = [s.scores(text) for s in self.systems]
        if self.method == "rrf":
            return sum(w / (self.rrf_k + _ranks(p)) for w, p in zip(self.weights, parts,
                                                                    strict=True))
        out = 0.0
        for w, p in zip(self.weights, parts, strict=True):
            span = p.max() - p.min()
            out = out + w * ((p - p.min()) / span if span > 0 else np.zeros_like(p))
        return out


class PassageSystem:
    """Document score from its best-matching *sentence* (sentence-level BM25, max per document).

    SciFact evidence is a few sentences inside an abstract, so a strong single sentence can
    be a better relevance signal than diluted whole-abstract term statistics.
    """

    def __init__(self, corpus: dict[int, Document], k1=1.2, b=0.75, stem=True, stop=True):
        self.stem, self.stop = stem, stop
        self.doc_ids = sorted(corpus)
        toks, starts = [], []
        for i in self.doc_ids:
            starts.append(len(toks))
            sents = corpus[i].sentences or ("",)
            toks.extend(tokenize(s, stem, stop) for s in sents)
        self.starts = np.asarray(starts)
        self.index = BM25Index(k1, b).fit(list(range(len(toks))), toks)

    def scores(self, text: str) -> np.ndarray:
        sent = self.index.score(tokenize(text, self.stem, self.stop))
        return np.maximum.reduceat(sent, self.starts)
