"""Search engine: document ranking plus sentence-level evidence highlighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from papermind_ingest import Document, tokenize
from papermind_retrieval.bm25 import BM25Index


@dataclass
class Hit:
    doc_id: int
    title: str
    score: float
    evidence_sentence: int
    evidence: str
    evidence_score: float


class SearchEngine:
    """BM25 over abstracts; the highest-scoring sentence of each hit is returned as evidence.

    Sentences are scored with a second, sentence-level BM25 index whose statistics come from
    the same corpus, so the highlighted sentence is the passage that best matches the query
    terms, not simply the first one.
    """

    def __init__(self, corpus: dict[int, Document], k1: float = 1.2, b: float = 0.75) -> None:
        if not corpus:
            raise ValueError("corpus is empty")
        self.corpus = corpus
        ids = sorted(corpus)
        self.docs = BM25Index(k1, b).fit(ids, [tokenize(corpus[i].text) for i in ids])
        sent_keys, sent_tokens = [], []
        for i in ids:
            for j, s in enumerate(corpus[i].sentences):
                sent_keys.append((i, j))
                sent_tokens.append(tokenize(s))
        self.sent_keys = sent_keys
        self.sent_tokens = sent_tokens
        self.sents = BM25Index(k1, b).fit(list(range(len(sent_keys))), sent_tokens)
        self._first_sentence = {}
        for pos, (d, _) in enumerate(sent_keys):
            self._first_sentence.setdefault(d, pos)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        q = tokenize(query)
        if not q:
            return []
        ids, scores = self.docs.search(q, k)
        sent_scores = self.sents.score(q)
        hits = []
        for d, s in zip(ids, scores, strict=True):
            if s <= 0:
                continue
            doc = self.corpus[int(d)]
            start = self._first_sentence[int(d)]
            local = sent_scores[start:start + len(doc.sentences)]
            j = int(np.argmax(local)) if len(local) else 0
            text = doc.sentences[j] if doc.sentences else ""
            hits.append(Hit(int(d), doc.title, float(s), j, text,
                            float(local[j]) if len(local) else 0.0))
        return hits
