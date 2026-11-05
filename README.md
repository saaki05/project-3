# PaperMind

Scientific-claim **evidence retrieval** on the SciFact benchmark: given a claim, find the
abstracts that support or contradict it, and show which sentence matters.

Three independently installable packages with one-way dependencies:

```
papermind-ingestion  ──▶  papermind-retrieval  ──▶  papermind-platform
 load/validate SciFact,    BM25 (from scratch), RM3,      FastAPI search service with
 text analysis, snapshot   LSA, neural dual-encoder,      sentence-level evidence
 (nltk only)               fusion, stats (torch)          highlighting
```

| Package | Responsibility |
|---|---|
| [`papermind-ingestion`](papermind-ingestion) | typed records, strict SciFact validation, tokenisation/stemming, checksummed corpus snapshot |
| [`papermind-retrieval`](papermind-retrieval) | retrievers, fusion, ranking metrics, bootstrap and permutation tests, the study runner |
| [`papermind-platform`](papermind-platform) | `/search`, `/documents/{id}`, `/health` over the snapshot |

## Research question

Strong lexical baselines are hard to beat on small scientific benchmarks. The study asks
which components genuinely help, with paired significance tests instead of single-number
comparisons: preprocessing (stemming, stop-words, title vs. abstract), BM25 parameter tuning,
pseudo-relevance feedback, latent semantic retrieval, a small trainable neural retriever, and
rank fusion, all tuned on training claims and reported on dev claims.

This is a controlled ablation study. It does not claim a new retrieval method.

## Status

| Item | State |
|---|---|
| All packages implemented, unit-tested (including tests against the real SciFact files) | yes |
| SciFact study | see `papermind-retrieval/results/REPORT.md`; if it is absent the study has not been run |
| Pretrained dense encoders (SPECTER, BGE, ...) | **not evaluated**: model hosting was unreachable from the development environment |
| Citation-graph reranking | **not tested**: SciFact has no citation graph |
| Database / vector store | not used (in-memory index) |

## Quick start

```bash
git clone https://github.com/saaki05/project-3 && cd project-3
pip install -e papermind-ingestion -e papermind-retrieval -e papermind-platform
# download SciFact (data.tar.gz) yourself and extract it, then:
papermind-retrieval study --data /path/to/scifact/data --out papermind-retrieval/results/scifact.json
papermind-retrieval report --results papermind-retrieval/results/scifact.json
papermind-ingest build --raw /path/to/scifact/data --out data/snapshot
papermind-serve --data data/snapshot
```

## Data and licenses

SciFact (Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020). Per the
upstream `LICENSE.md` when this was written: claims and annotations CC BY 4.0, abstracts
(from S2ORC) ODC-By 1.0. Verify upstream before redistributing; no data is included here.

## License

MIT.
