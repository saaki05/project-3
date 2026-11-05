# papermind-platform

HTTP search service over a scientific-abstract corpus. For each query it returns ranked
abstracts and, for every hit, the sentence that best matches the query terms.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | corpus size and BM25 parameters |
| `GET /search?q=...&k=10` | ranked results: `doc_id`, `title`, `score`, `evidence_sentence`, `evidence`, `evidence_score` |
| `GET /documents/{doc_id}` | title and sentences of one abstract |

Queries are validated (non-empty, `1 <= k <= 100`); unknown documents return 404.

## Install, test, run

```bash
git clone https://github.com/saaki05/project-3 && cd project-3
pip install -e papermind-ingestion -e papermind-retrieval
cd papermind-platform && pip install -e ".[dev]" && pytest

papermind-ingest build --raw /path/to/scifact/data --out data/snapshot
papermind-serve --data data/snapshot --port 8000
curl "http://127.0.0.1:8000/search?q=statins%20reduce%20cholesterol&k=3"
```

Use the `k1`/`b` values found by `papermind-retrieval study` via `--k1` and `--b`.

## Design notes and limitations

- The index is built in memory at start-up (seconds for the 5k-abstract SciFact corpus).
  There is no database: a persistent store (PostgreSQL/pgvector) would only be justified
  once a dense retriever and a much larger corpus are added.
- Evidence sentences come from a sentence-level BM25 index, so they reflect lexical overlap
  and not semantic entailment. They are highlights to help a reader, not verified evidence.
- No authentication or rate limiting; run it behind a gateway if exposed.

## License

MIT.
