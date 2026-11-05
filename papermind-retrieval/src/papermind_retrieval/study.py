"""The full SciFact evidence-retrieval study.

Protocol: every hyper-parameter is chosen on the *training* claims; all reported numbers
come from the *dev* claims (the public test claims have no evidence labels). Retrieval
metrics are computed over claims with at least one relevant abstract.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from papermind_ingest import Claim, load_scifact

from papermind_retrieval import evaluation as E
from papermind_retrieval.neural import HashedEncoder, NeuralSystem, train_pairs
from papermind_retrieval.systems import (
    BM25System,
    FusionSystem,
    LSASystem,
    PassageSystem,
    RM3System,
)


def run_system(system, claims: list[Claim], doc_ids: list[int], k: int = 100):
    return {c.claim_id: E.rank_docs(system.scores(c.text), doc_ids, k) for c in claims}


def evaluate_system(system, claims, doc_ids) -> dict[str, np.ndarray]:
    return E.per_query_metrics(run_system(system, claims, doc_ids), claims)


def _mean(system, claims, doc_ids, metric="ndcg@10") -> float:
    return float(evaluate_system(system, claims, doc_ids)[metric].mean())


def _summaries(per_query: dict[str, np.ndarray]) -> dict:
    out = {}
    for m, v in per_query.items():
        mean, lo, hi = E.bootstrap_mean_ci(v)
        out[m] = {"mean": mean, "ci95": [lo, hi]}
    return out


def _versus(per_query, baseline) -> dict:
    return {m: {**E.paired_bootstrap(per_query[m], baseline[m]),
                "perm_p": E.paired_permutation(per_query[m], baseline[m])}
            for m in ("ndcg@10", "recall@100")}


def run_study(data_dir, out_path=None, seeds=(0, 1, 2), pretrain_epochs=15, finetune_epochs=15):
    bench = load_scifact(data_dir)
    corpus = bench.corpus
    doc_ids = sorted(corpus)
    train = [c for c in bench.train if c.has_evidence]
    dev = bench.dev
    result: dict = {"n_docs": len(doc_ids), "n_train_claims_with_evidence": len(train),
                    "n_dev_claims_with_evidence": sum(c.has_evidence for c in dev),
                    "seeds": list(seeds), "tuning": {}, "systems": {}, "ablations": {},
                    "errors": {}}

    # ---- baselines and preprocessing/field ablations (default parameters, no tuning)
    default = BM25System(corpus)
    base_pq = evaluate_system(default, dev, doc_ids)
    result["systems"]["bm25 (k1=1.2, b=0.75)"] = {"metrics": _summaries(base_pq)}
    for name, kwargs in {"no stemming": {"stem": False}, "no stop-words": {"stop": False},
                         "title only": {"field": "title"},
                         "abstract only": {"field": "abstract"}}.items():
        pq = evaluate_system(BM25System(corpus, **kwargs), dev, doc_ids)
        result["ablations"][name] = {"metrics": _summaries(pq), "vs_default": _versus(pq, base_pq)}

    # ---- BM25 parameter tuning on train claims
    grid = list(itertools.product([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9]))
    scores = {}
    tuner = BM25System(corpus)
    for k1, b in grid:
        scores[(k1, b)] = _mean(tuner.set_params(k1, b), train, doc_ids)
    (k1, b), best = max(scores.items(), key=lambda kv: kv[1])
    result["tuning"]["bm25"] = {"best": {"k1": k1, "b": b, "train_ndcg@10": best},
                                "default_train_ndcg@10": scores[(1.2, 0.75)],
                                "grid": {f"{a},{c}": v for (a, c), v in scores.items()}}
    tuned = BM25System(corpus, k1, b)
    tuned_pq = evaluate_system(tuned, dev, doc_ids)
    result["systems"]["bm25 tuned"] = {"metrics": _summaries(tuned_pq),
                                       "vs_default": _versus(tuned_pq, base_pq),
                                       "params": {"k1": k1, "b": b}}

    # ---- pseudo-relevance feedback
    rm_scores = {}
    for fd, ft, lam in itertools.product([3, 5, 10], [5, 10, 20], [0.4, 0.6, 0.8]):
        rm_scores[(fd, ft, lam)] = _mean(RM3System(tuned, fd, ft, lam), train, doc_ids)
    (fd, ft, lam), best_rm = max(rm_scores.items(), key=lambda kv: kv[1])
    result["tuning"]["rm3"] = {"best": {"fb_docs": fd, "fb_terms": ft, "lam": lam,
                                        "train_ndcg@10": best_rm}}
    rm3 = RM3System(tuned, fd, ft, lam)
    rm3_pq = evaluate_system(rm3, dev, doc_ids)
    result["systems"]["bm25 tuned + RM3"] = {"metrics": _summaries(rm3_pq),
                                             "vs_default": _versus(rm3_pq, base_pq)}

    # ---- latent semantic analysis
    lsa_scores = {}
    for dims in (64, 128, 256, 512):
        lsa_scores[dims] = _mean(LSASystem(corpus, dims), train, doc_ids)
    best_dims = max(lsa_scores, key=lsa_scores.get)
    lsa = LSASystem(corpus, best_dims)
    lsa_pq = evaluate_system(lsa, dev, doc_ids)
    result["tuning"]["lsa"] = {"best_dims": best_dims, "train_ndcg@10": lsa_scores}
    result["systems"]["LSA"] = {"metrics": _summaries(lsa_pq),
                                "vs_default": _versus(lsa_pq, base_pq)}

    # ---- passage-level evidence scoring, alone and mixed with document BM25
    passage = PassageSystem(corpus, k1, b)
    pass_pq = evaluate_system(passage, dev, doc_ids)
    result["systems"]["best-sentence BM25"] = {"metrics": _summaries(pass_pq),
                                               "vs_default": _versus(pass_pq, base_pq)}
    w_scores = {}
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        w_scores[w] = _mean(FusionSystem([tuned, passage], "minmax", [1 - w, w]), train, doc_ids)
    best_w = max(w_scores, key=w_scores.get)
    mixed = FusionSystem([tuned, passage], "minmax", [1 - best_w, best_w])
    mixed_pq = evaluate_system(mixed, dev, doc_ids)
    result["tuning"]["passage_mix"] = {"best_weight_on_passage": best_w,
                                       "train_ndcg@10": {str(k): v for k, v in w_scores.items()}}
    result["systems"]["BM25 + best-sentence (tuned mix)"] = {
        "metrics": _summaries(mixed_pq), "vs_default": _versus(mixed_pq, base_pq),
        "params": {"weight_on_passage": best_w}}

    # ---- fusion of lexical and latent
    fused = {"RRF(bm25 tuned, LSA)": FusionSystem([tuned, lsa], "rrf"),
             "min-max(bm25 tuned, LSA)": FusionSystem([tuned, lsa], "minmax")}
    for name, system in fused.items():
        pq = evaluate_system(system, dev, doc_ids)
        result["systems"][name] = {"metrics": _summaries(pq), "vs_default": _versus(pq, base_pq)}

    # ---- trainable neural retriever (multiple seeds)
    doc_texts = [corpus[i].text for i in doc_ids]
    pre_pairs = [(corpus[i].title, corpus[i].abstract) for i in doc_ids if corpus[i].title]
    ft_pairs = [(c.text, corpus[d].text) for c in train for d in c.relevant_docs]
    per_seed: dict[str, list[dict[str, np.ndarray]]] = {"neural (pretrained only)": [],
                                                        "neural (+ fine-tuned)": [],
                                                        "RRF(bm25 tuned, neural)": [],
                                                        "RRF(bm25 tuned, LSA, neural)": []}
    for seed in seeds:
        torch_model = HashedEncoder(seed=seed)
        train_pairs(torch_model, pre_pairs, epochs=pretrain_epochs, seed=seed)
        pre = NeuralSystem(torch_model, doc_texts)
        per_seed["neural (pretrained only)"].append(evaluate_system(pre, dev, doc_ids))
        train_pairs(torch_model, ft_pairs, epochs=finetune_epochs, lr=1e-3, seed=seed)
        neural = NeuralSystem(torch_model, doc_texts)
        per_seed["neural (+ fine-tuned)"].append(evaluate_system(neural, dev, doc_ids))
        per_seed["RRF(bm25 tuned, neural)"].append(
            evaluate_system(FusionSystem([tuned, neural], "rrf"), dev, doc_ids))
        per_seed["RRF(bm25 tuned, LSA, neural)"].append(
            evaluate_system(FusionSystem([tuned, lsa, neural], "rrf"), dev, doc_ids))
    for name, runs in per_seed.items():
        avg = {m: np.mean([r[m] for r in runs], axis=0) for m in runs[0]}
        spread = {m: float(np.std([r[m].mean() for r in runs])) for m in runs[0]}
        result["systems"][name] = {"metrics": _summaries(avg), "seed_std": spread,
                                   "vs_default": _versus(avg, base_pq),
                                   "note": "per-query metrics averaged over seeds"}

    # ---- error analysis of the strongest system by nDCG@10
    best_name = max(result["systems"],
                    key=lambda n: result["systems"][n]["metrics"]["ndcg@10"]["mean"])
    best_sys = {"bm25 tuned": tuned, "bm25 tuned + RM3": rm3, "LSA": lsa, "best-sentence BM25":
                passage, "BM25 + best-sentence (tuned mix)": mixed}.get(best_name) \
        or fused.get(best_name) or tuned
    run = run_system(best_sys, dev, doc_ids)
    miss10, miss100, by_n = [], [], {}
    for c in dev:
        if not c.has_evidence:
            continue
        r = run[c.claim_id]
        hit10 = bool(c.relevant_docs & set(r[:10]))
        hit100 = bool(c.relevant_docs & set(r[:100]))
        by_n.setdefault(len(c.relevant_docs), []).append(hit10)
        if not hit100:
            miss100.append({"claim_id": c.claim_id, "claim": c.text,
                            "relevant_titles": [corpus[d].title for d in c.relevant_docs]})
        elif not hit10:
            miss10.append(c.claim_id)
    result["errors"] = {
        "analysed_system": best_name if best_name in (
            "bm25 tuned", "bm25 tuned + RM3", "LSA", "best-sentence BM25",
            "BM25 + best-sentence (tuned mix)") or best_name in fused
        else "bm25 tuned (neural systems not re-ranked here)",
        "claims_missed_at_10_but_found_at_100": len(miss10),
        "claims_missed_at_100": len(miss100),
        "hit_rate@10_by_num_relevant_docs": {str(k): float(np.mean(v))
                                             for k, v in sorted(by_n.items())},
        "examples_missed_at_100": miss100[:10],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result
