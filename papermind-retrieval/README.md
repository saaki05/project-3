# papermind-retrieval

Retrieval algorithms and the evaluation study for PaperMind: **which components actually
improve evidence retrieval for scientific claims, and by how much, measured with paired
significance tests?**

## Implemented

| Component | Notes |
|---|---|
| `BM25Index` | Okapi BM25 over a sparse tf matrix, written from scratch; Lucene-style non-negative idf; weights recomputed per `(k1, b)` so sweeps are cheap; save/load without pickle |
| `RM3System` | pseudo-relevance feedback: score-weighted term distribution of the top documents interpolated with the query |
| `LSASystem` | sublinear TF-IDF + truncated SVD, cosine retrieval |
| `HashedEncoder` | compact trainable dual-encoder over hashed unigram+bigram features, contrastive InfoNCE training with in-batch negatives; seeded initialisation |
| `FusionSystem` | reciprocal rank fusion and weighted min-max score fusion |
| Evaluation | nDCG@10, MRR@10, Recall@10/100; bootstrap CIs; paired bootstrap and sign-flip permutation tests |

## Study protocol (SciFact)

- All hyper-parameters (BM25 `k1`/`b`, RM3 settings, LSA dimension) are chosen on the
  **training claims**; every reported number is on the **dev claims** (the public test claims
  have no labels). Metrics are computed over claims with at least one relevant abstract.
- Ablations at default BM25 settings: no stemming, no stop-word removal, title-only,
  abstract-only.
- The neural retriever is pretrained on title→abstract pairs of the corpus (no claim labels
  used) and then fine-tuned on training-claim → cited-abstract pairs, for several seeds;
  per-query metrics are averaged over seeds before testing.
- Error analysis: claims missed at 10 / 100, hit rate by number of relevant abstracts.

## Install, test, run

```bash
git clone https://github.com/saaki05/project-3 && cd project-3
pip install -e papermind-ingestion
cd papermind-retrieval && pip install -e ".[dev]" && pytest

papermind-retrieval study --data /path/to/scifact/data --out results/scifact.json
papermind-retrieval report --results results/scifact.json --out results/REPORT.md
```

## Results

`results/REPORT.md` is generated from `results/scifact.json`. If it is absent the study has
not been run in the environment you are looking at.

## Scope and limitations

- No pretrained transformer encoder (SPECTER, BGE, ...) is evaluated: model hosting was not
  reachable from the development environment. The `HashedEncoder` is a lightweight stand-in
  and is not expected to match pretrained models; a pretrained dense retriever is the obvious
  next system to add to `FusionSystem`.
- SciFact has no citation graph, so graph-based reranking is not tested here.
- Claim verification (support / refute) is out of scope; only evidence retrieval is studied.
- Dev has a few hundred claims, so differences of one or two nDCG points are usually within
  the reported confidence intervals.

## License

MIT.
