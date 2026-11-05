"""Normalised corpus snapshot with a checksum manifest (the contract with downstream repos)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from papermind_ingest.schema import Document

SCHEMA_VERSION = 1


def write_corpus(corpus: dict[int, Document], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for d in sorted(corpus.values(), key=lambda d: d.doc_id):
            fh.write(json.dumps({"doc_id": d.doc_id, "title": d.title,
                                 "sentences": list(d.sentences)}, ensure_ascii=False) + "\n")
    manifest = {"schema_version": SCHEMA_VERSION, "num_documents": len(corpus),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_corpus(data_dir: str | Path, verify: bool = True) -> dict[int, Document]:
    d = Path(data_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {manifest['schema_version']}")
    if verify:
        digest = hashlib.sha256((d / "corpus.jsonl").read_bytes()).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError("checksum mismatch for corpus.jsonl")
    corpus = {}
    for line in (d / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        corpus[rec["doc_id"]] = Document(rec["doc_id"], rec["title"], tuple(rec["sentences"]))
    return corpus
