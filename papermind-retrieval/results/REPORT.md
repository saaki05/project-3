# PaperMind retrieval results on SciFact dev (generated from the result JSON)

Corpus: 5183 abstracts. Dev claims with evidence: 188. Hyper-parameters tuned on 505 training claims. Seeds: [0, 1, 2].

| system | nDCG@10 [95% CI] | MRR@10 | Recall@10 | Recall@100 | Δ nDCG@10 vs default (paired bootstrap p) |
|---|---|---|---|---|---|
| bm25 (k1=1.2, b=0.75) | 0.833 [0.792, 0.872] | 0.801 | 0.941 | 0.972 | baseline |
| bm25 tuned | 0.835 [0.793, 0.874] | 0.805 | 0.937 | 0.972 | +0.002 (p=0.705) |
| bm25 tuned + RM3 | 0.832 [0.790, 0.873] | 0.805 | 0.931 | 0.980 | -0.001 (p=0.902) |
| LSA | 0.684 [0.631, 0.734] | 0.640 | 0.842 | 0.979 | -0.149 (p=0.000) |
| best-sentence BM25 | 0.726 [0.673, 0.776] | 0.688 | 0.865 | 0.966 | -0.107 (p=0.000) |
| BM25 + best-sentence (tuned mix) | 0.835 [0.793, 0.874] | 0.805 | 0.937 | 0.972 | +0.002 (p=0.705) |
| RRF(bm25 tuned, LSA) | 0.753 [0.708, 0.797] | 0.702 | 0.924 | 0.983 | -0.080 (p=0.000) |
| min-max(bm25 tuned, LSA) | 0.824 [0.784, 0.863] | 0.792 | 0.943 | 0.988 | -0.009 (p=0.367) |
| neural (pretrained only) | 0.392 [0.342, 0.444] | 0.355 | 0.525 | 0.739 | -0.441 (p=0.000) |
| neural (+ fine-tuned) | 0.634 [0.580, 0.689] | 0.608 | 0.728 | 0.851 | -0.199 (p=0.000) |
| RRF(bm25 tuned, neural) | 0.762 [0.714, 0.807] | 0.738 | 0.863 | 0.986 | -0.071 (p=0.000) |
| RRF(bm25 tuned, LSA, neural) | 0.765 [0.722, 0.807] | 0.727 | 0.909 | 0.990 | -0.067 (p=0.000) |

## Preprocessing and field ablations (default BM25 parameters)

| variant | nDCG@10 [95% CI] | MRR@10 | Recall@10 | Recall@100 | Δ nDCG@10 vs default |
|---|---|---|---|---|---|
| no stemming | 0.819 [0.776, 0.863] | 0.791 | 0.921 | 0.954 | -0.014 (p=0.137) |
| no stop-words | 0.834 [0.793, 0.875] | 0.803 | 0.938 | 0.974 | +0.001 (p=0.810) |
| title only | 0.574 [0.521, 0.629] | 0.527 | 0.734 | 0.871 | -0.259 (p=0.000) |
| abstract only | 0.822 [0.782, 0.863] | 0.790 | 0.931 | 0.972 | -0.010 (p=0.020) |

## Tuning (on training claims)

- BM25: best k1=1.5, b=0.5 (train nDCG@10 0.859 vs default 0.855)
- RM3: {'fb_docs': 10, 'fb_terms': 20, 'lam': 0.8, 'train_ndcg@10': 0.8488660541089059}
- LSA: best dims 512
- passage mix: weight on best-sentence score 0.0

## Error analysis

```
{
  "analysed_system": "bm25 tuned",
  "claims_missed_at_10_but_found_at_100": 6,
  "claims_missed_at_100": 4,
  "hit_rate@10_by_num_relevant_docs": {
    "1": 0.949438202247191,
    "2": 0.75,
    "3": 1.0,
    "4": 1.0,
    "5": 1.0
  }
}
```
