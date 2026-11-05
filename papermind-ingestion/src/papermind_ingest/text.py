"""Tokenisation and normalisation shared by every retriever."""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_TOKEN = re.compile(r"[a-z0-9]+")

# A short, conventional English stop list (function words only).
STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being "
    "below between both but by can could did do does doing down during each few for from further "
    "had has have having he her here hers him his how i if in into is it its itself just me more "
    "most my no nor not of off on once only or other our out over own same she should so some such "
    "than that the their them then there these they this those through to too under until up very "
    "was we were what when where which while who whom why will with would you your".split()
)

_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    return _stemmer.stem(token)


def tokenize(text: str, stem: bool = True, stop: bool = True) -> list[str]:
    """Lower-case alphanumeric tokens, optionally stop-word filtered and Porter-stemmed."""
    toks = _TOKEN.findall(text.lower())
    if stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks] if stem else toks
