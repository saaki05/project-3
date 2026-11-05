"""Loader and validator for the SciFact release layout.

Files: ``corpus.jsonl`` (``doc_id, title, abstract[list of sentences]``) and
``claims_{train,dev,test}.jsonl`` (``id, claim, evidence{doc_id: [{sentences, label}]}``,
``cited_doc_ids``). The test claims carry no evidence labels in the public release.

Licenses (verify against the upstream repository before redistribution): claims and
annotations CC BY 4.0; abstracts are part of S2ORC, ODC-By 1.0.
Reference: Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020.
"""

from __future__ import annotations

import json
from pathlib import Path

from papermind_ingest.schema import LABELS, Benchmark, Claim, Document, Evidence


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as err:
                    raise ValueError(f"{path.name}:{line_no}: invalid JSON") from err


def load_corpus(path: str | Path) -> dict[int, Document]:
    corpus: dict[int, Document] = {}
    for rec in _read_jsonl(Path(path)):
        doc = Document(int(rec["doc_id"]), rec.get("title", "").strip(),
                       tuple(s.strip() for s in rec["abstract"]))
        if doc.doc_id in corpus:
            raise ValueError(f"duplicate doc_id {doc.doc_id}")
        corpus[doc.doc_id] = doc
    return corpus


def load_claims(path: str | Path) -> list[Claim]:
    claims, seen = [], set()
    for rec in _read_jsonl(Path(path)):
        cid = int(rec["id"])
        if cid in seen:
            raise ValueError(f"duplicate claim id {cid}")
        seen.add(cid)
        evidence = {
            int(doc): tuple(Evidence(e["label"], tuple(e["sentences"])) for e in items)
            for doc, items in rec.get("evidence", {}).items()
        }
        claims.append(Claim(cid, rec["claim"].strip(), evidence,
                            tuple(int(d) for d in rec.get("cited_doc_ids", []))))
    return claims


def validate(corpus: dict[int, Document], claims: list[Claim]) -> None:
    """Every evidence document must exist, labels must be known, sentence indices in range."""
    for c in claims:
        for doc_id, items in c.evidence.items():
            if doc_id not in corpus:
                raise ValueError(f"claim {c.claim_id}: evidence doc {doc_id} not in corpus")
            for e in items:
                if e.label not in LABELS:
                    raise ValueError(f"claim {c.claim_id}: unknown label {e.label!r}")
                n = len(corpus[doc_id].sentences)
                if any(not 0 <= s < n for s in e.sentences):
                    raise ValueError(f"claim {c.claim_id}: sentence index out of range")


def load_scifact(data_dir: str | Path) -> Benchmark:
    d = Path(data_dir)
    corpus = load_corpus(d / "corpus.jsonl")
    bench = Benchmark(corpus, load_claims(d / "claims_train.jsonl"),
                      load_claims(d / "claims_dev.jsonl"), load_claims(d / "claims_test.jsonl"))
    validate(corpus, bench.train)
    validate(corpus, bench.dev)
    return bench
