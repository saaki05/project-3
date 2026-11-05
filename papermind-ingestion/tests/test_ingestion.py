import json
import os

import pytest

from papermind_ingest import (
    Document,
    load_scifact,
    read_corpus,
    tokenize,
    validate,
    write_corpus,
)
from papermind_ingest.cli import main
from papermind_ingest.scifact import load_claims


def write_fake(root):
    docs = [
        {"doc_id": 1, "title": "Cats", "abstract": ["Cats purr.", "Cats sleep a lot."]},
        {"doc_id": 2, "title": "Dogs", "abstract": ["Dogs bark.", "Dogs fetch sticks."]},
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    train = [{"id": 10, "claim": "Cats purr.", "evidence":
              {"1": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [1]},
             {"id": 11, "claim": "Birds fly.", "evidence": {}, "cited_doc_ids": []}]
    (root / "claims_train.jsonl").write_text("\n".join(json.dumps(c) for c in train))
    (root / "claims_dev.jsonl").write_text(json.dumps(
        {"id": 20, "claim": "Dogs bark.", "evidence":
         {"2": [{"sentences": [0, 1], "label": "SUPPORT"}]}, "cited_doc_ids": [2]}))
    (root / "claims_test.jsonl").write_text(json.dumps({"id": 30, "claim": "Fish swim."}))


def test_loader_and_properties(tmp_path):
    write_fake(tmp_path)
    b = load_scifact(tmp_path)
    assert len(b.corpus) == 2 and len(b.train) == 2 and len(b.test) == 1
    c = b.train[0]
    assert c.relevant_docs == frozenset({1}) and c.has_evidence
    assert not b.train[1].has_evidence and not b.test[0].has_evidence
    assert b.corpus[1].text == "Cats. Cats purr. Cats sleep a lot."


def test_validation_catches_bad_references(tmp_path):
    write_fake(tmp_path)
    b = load_scifact(tmp_path)
    bad = [{"id": 99, "claim": "x", "evidence": {"7": [{"sentences": [0], "label": "SUPPORT"}]}}]
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad[0]))
    with pytest.raises(ValueError, match="not in corpus"):
        validate(b.corpus, load_claims(tmp_path / "bad.jsonl"))
    bad[0]["evidence"] = {"1": [{"sentences": [5], "label": "SUPPORT"}]}
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad[0]))
    with pytest.raises(ValueError, match="out of range"):
        validate(b.corpus, load_claims(tmp_path / "bad.jsonl"))
    bad[0]["evidence"] = {"1": [{"sentences": [0], "label": "MAYBE"}]}
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad[0]))
    with pytest.raises(ValueError, match="unknown label"):
        validate(b.corpus, load_claims(tmp_path / "bad.jsonl"))


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "c.jsonl").write_text(
        json.dumps({"id": 1, "claim": "a"}) + "\n" + json.dumps({"id": 1, "claim": "b"}))
    with pytest.raises(ValueError, match="duplicate"):
        load_claims(tmp_path / "c.jsonl")


def test_tokenize_stems_and_filters():
    assert tokenize("The Cats are purring!") == ["cat", "pur"]
    assert tokenize("The Cats", stem=False, stop=False) == ["the", "cats"]
    assert tokenize("IL-6 inhibitors, 10% of patients") == ["il", "6", "inhibitor", "10", "patient"]
    assert tokenize("") == []


def test_store_round_trip_and_checksum(tmp_path):
    corpus = {1: Document(1, "T", ("a b.", "c d."))}
    write_corpus(corpus, tmp_path)
    assert read_corpus(tmp_path) == corpus
    (tmp_path / "corpus.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        read_corpus(tmp_path)


def test_cli(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fake(raw)
    assert main(["stats", "--raw", str(raw)]) == 0
    assert "train: 2 claims, 1 with evidence" in capsys.readouterr().out
    assert main(["build", "--raw", str(raw), "--out", str(tmp_path / "out")]) == 0


@pytest.mark.skipif("SCIFACT_DIR" not in os.environ, reason="real SciFact data not set")
def test_real_scifact():
    b = load_scifact(os.environ["SCIFACT_DIR"])
    assert len(b.corpus) == 5183 and len(b.train) == 809
    assert len(b.dev) == 300 and len(b.test) == 300
    assert sum(c.has_evidence for c in b.dev) > 100
