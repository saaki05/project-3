"""FastAPI application: search, document lookup and health."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from papermind_ingest import Document

from papermind_platform.engine import SearchEngine


def create_app(corpus: dict[int, Document], k1: float = 1.2, b: float = 0.75) -> FastAPI:
    engine = SearchEngine(corpus, k1, b)
    app = FastAPI(title="PaperMind", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(corpus), "k1": k1, "b": b}

    @app.get("/search")
    def search(q: str = Query(..., min_length=1, max_length=1000),
               k: int = Query(10, ge=1, le=100)) -> dict:
        hits = engine.search(q, k)
        return {"query": q, "num_results": len(hits), "results": [asdict(h) for h in hits]}

    @app.get("/documents/{doc_id}")
    def document(doc_id: int) -> dict:
        doc = corpus.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return {"doc_id": doc.doc_id, "title": doc.title, "sentences": list(doc.sentences)}

    return app
