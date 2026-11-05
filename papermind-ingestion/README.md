# papermind-ingestion

Corpus and claim layer for PaperMind (scientific-claim evidence retrieval).

## What it does

- Typed records: `Document` (id, title, sentences), `Claim` (id, text, evidence per
  abstract with label and rationale sentence indices), `Benchmark`.
- **SciFact loader and validator**: duplicate ids, evidence documents missing from the
  corpus, unknown labels and out-of-range sentence indices are errors. The public test claims
  have no evidence labels; the loader represents them as claims with empty evidence.
- **Text analysis** shared by every retriever: lower-casing, alphanumeric tokens, a function-
  word stop list and Porter stemming (NLTK's `PorterStemmer`, which needs no data download).
- A checksummed normalised corpus snapshot (`corpus.jsonl` + `manifest.json`), the contract
  with `papermind-platform`.

## Install and test

```bash
git clone https://github.com/saaki05/project-3 && cd project-3/papermind-ingestion
pip install -e ".[dev]" && pytest
SCIFACT_DIR=/path/to/scifact/data pytest      # also checks the real SciFact counts
```

## Data

SciFact (Wadden et al., EMNLP 2020), `data.tar.gz` from the Allen AI release. Licenses per
the upstream `LICENSE.md` at the time of writing: claims and annotations CC BY 4.0, abstracts
(part of S2ORC) ODC-By 1.0; confirm before redistributing. Not included in this repository.

```bash
papermind-ingest stats --raw /path/to/scifact/data
papermind-ingest build --raw /path/to/scifact/data --out data/snapshot
```

## License

MIT.
