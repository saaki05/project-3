import pytest
from fastapi.testclient import TestClient
from papermind_ingest import Document

from papermind_platform.api import create_app
from papermind_platform.engine import SearchEngine


@pytest.fixture
def corpus():
    return {
        1: Document(1, "Statins and heart disease", (
            "Cholesterol is a risk factor.", "Statins lower LDL cholesterol in patients.",
            "Trial enrolled adults.")),
        2: Document(2, "Memory and the hippocampus", (
            "The hippocampus encodes episodic memory.", "Lesions impair recall.")),
        3: Document(3, "Genome sequencing methods", ("Sequencing reads were aligned.",)),
    }


def test_engine_ranks_and_highlights_the_best_sentence(corpus):
    eng = SearchEngine(corpus)
    hits = eng.search("statins lower cholesterol", k=3)
    assert hits[0].doc_id == 1 and hits[0].evidence_sentence == 1
    assert "Statins lower LDL" in hits[0].evidence
    assert [h.doc_id for h in eng.search("hippocampus memory")][0] == 2
    assert eng.search("the of and") == []  # only stop-words
    assert eng.search("zebra") == []
    with pytest.raises(ValueError):
        eng.search("x", k=0)
    with pytest.raises(ValueError):
        SearchEngine({})


def test_api_endpoints_and_validation(corpus):
    client = TestClient(create_app(corpus))
    assert client.get("/health").json()["documents"] == 3
    r = client.get("/search", params={"q": "lesions recall", "k": 2})
    assert r.status_code == 200 and r.json()["results"][0]["doc_id"] == 2
    assert client.get("/documents/3").json()["title"] == "Genome sequencing methods"
    assert client.get("/documents/99").status_code == 404
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 500}).status_code == 422
