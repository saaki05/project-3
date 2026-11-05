import json
import math

import numpy as np
import pytest
from papermind_ingest import Claim, Document, Evidence

from papermind_retrieval import evaluation as E
from papermind_retrieval.bm25 import BM25Index
from papermind_retrieval.cli import main, render
from papermind_retrieval.neural import HashedEncoder, NeuralSystem, hashed_ids, train_pairs
from papermind_retrieval.study import run_study
from papermind_retrieval.systems import BM25System, FusionSystem, LSASystem, RM3System


def test_bm25_matches_hand_computation():
    docs = [["a", "b", "b"], ["a", "c"], ["d"]]
    idx = BM25Index(k1=1.2, b=0.75).fit([10, 20, 30], docs)
    n, df_b, avgdl = 3, 1, (3 + 2 + 1) / 3
    idf = math.log(1 + (n - df_b + 0.5) / (df_b + 0.5))
    tf, dl = 2, 3
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    s = idx.score(["b"])
    assert s[0] == pytest.approx(expected, rel=1e-5)
    assert s[1] == 0 and s[2] == 0
    ids, _ = idx.search(["b"], k=2)
    assert ids[0] == 10


def test_bm25_properties():
    idx = BM25Index().fit([1, 2, 3], [["x", "y"], ["x"], ["z"]])
    assert idx.score(["unseen"]).sum() == 0
    assert np.allclose(idx.score(["x", "x"]), 2 * idx.score(["x"]))  # query tf counted
    shorter_wins = idx.score(["x"])
    assert shorter_wins[1] > shorter_wins[0]  # same tf, shorter document scores higher
    idx.set_params(1.2, 0.0)
    assert idx.score(["x"])[1] == pytest.approx(idx.score(["x"])[0])  # no length normalisation
    assert (idx.score(["x", "y", "z"]) >= 0).all()


def test_bm25_save_load_round_trip(tmp_path):
    idx = BM25Index(1.5, 0.6).fit([5, 6], [["a", "b"], ["b", "c", "c"]])
    idx.save(tmp_path)
    back = BM25Index.load(tmp_path)
    assert np.allclose(back.score(["c", "b"]), idx.score(["c", "b"]))
    assert (back.k1, back.b) == (1.5, 0.6)


def corpus_fixture():
    topics = {"cardio": "heart cardiac artery blood pressure",
              "neuro": "brain neuron cortex memory",
              "gene": "gene dna mutation genome protein"}
    corpus = {}
    for t_i, (name, words) in enumerate(topics.items()):
        for j in range(8):
            did = t_i * 100 + j
            corpus[did] = Document(did, f"{name} study {j}",
                                   (f"{words} observation {j}.", f"{words} result"))
    return corpus, topics


def test_systems_retrieve_the_right_topic():
    corpus, topics = corpus_fixture()
    ids = sorted(corpus)
    bm25 = BM25System(corpus)
    lsa = LSASystem(corpus, dims=5)
    fused = FusionSystem([bm25, lsa], "rrf")
    rm3 = RM3System(bm25, fb_docs=3, fb_terms=5, lam=0.5)
    for system in (bm25, lsa, fused, rm3, FusionSystem([bm25, lsa], "minmax")):
        top = E.rank_docs(system.scores("brain memory cortex"), ids, 5)
        assert all(100 <= d < 200 for d in top[:3]), type(system).__name__
    assert all(200 <= d < 300 for d in E.rank_docs(bm25.scores("dna mutation"), ids, 3))


def test_ranking_metrics_hand_values():
    rel = frozenset({1, 2})
    ranked = [9, 1, 8, 2]
    assert E.mrr_at_k(ranked, rel, 10) == 0.5
    assert E.recall_at_k(ranked, rel, 2) == 0.5 and E.recall_at_k(ranked, rel, 4) == 1.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert E.ndcg_at_k(ranked, rel, 10) == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert E.ndcg_at_k([1, 2], rel, 10) == pytest.approx(1.0)
    assert E.ndcg_at_k([], frozenset(), 10) == 0.0


def test_per_query_metrics_skip_claims_without_evidence():
    c1 = Claim(1, "x", {5: (Evidence("SUPPORT", (0,)),)})
    c2 = Claim(2, "y")
    pq = E.per_query_metrics({1: [5], 2: [3]}, [c1, c2])
    assert len(pq["ndcg@10"]) == 1 and pq["ndcg@10"][0] == 1.0


def test_significance_tests():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.4, 0.9, 200)
    b = a - 0.1 + rng.normal(scale=0.05, size=200)
    res = E.paired_bootstrap(a, b)
    assert res["diff"] > 0.05 and res["ci_low"] > 0 and res["p_value"] < 0.01
    assert E.paired_permutation(a, b) < 0.01
    same = E.paired_bootstrap(a, a)
    assert same["diff"] == 0 and E.paired_permutation(a, a) > 0.9
    with pytest.raises(ValueError):
        E.paired_bootstrap(a, b[:-1])
    mean, lo, hi = E.bootstrap_mean_ci(a)
    assert lo <= mean <= hi


def test_hashing_is_stable_and_bounded():
    ids = hashed_ids("Heart disease risk factors", 1 << 10)
    assert ids == hashed_ids("Heart disease risk factors", 1 << 10)
    assert all(0 <= i < 1 << 10 for i in ids)
    assert hashed_ids("the", 100) == [0]  # stop-word only text falls back to bucket 0


def test_neural_encoder_learns_pairs():
    corpus, topics = corpus_fixture()
    pairs = [(f"{w} question", corpus[did].text) for did in corpus
             for w in [topics[["cardio", "neuro", "gene"][did // 100]]]]
    model = HashedEncoder(buckets=1 << 10, dim=16)
    losses = train_pairs(model, pairs, epochs=25, batch=8, lr=1e-2, seed=0)
    assert losses[-1] < losses[0]
    ids = sorted(corpus)
    system = NeuralSystem(model, [corpus[i].text for i in ids])
    top = E.rank_docs(system.scores("brain neuron cortex memory"), ids, 3)
    assert all(100 <= d < 200 for d in top)
    # same seed, same result
    m2 = HashedEncoder(buckets=1 << 10, dim=16)
    torch_state = train_pairs(m2, pairs, epochs=25, batch=8, lr=1e-2, seed=0)
    assert torch_state == pytest.approx(losses)


def write_scifact(root):
    corpus, topics = corpus_fixture()
    (root / "corpus.jsonl").write_text("\n".join(
        json.dumps({"doc_id": d.doc_id, "title": d.title, "abstract": list(d.sentences)})
        for d in corpus.values()))

    def claims(prefix, start, n):
        out = []
        for i in range(n):
            t = i % 3
            words = list(topics.values())[t].split()
            did = t * 100 + (i % 8)
            out.append({"id": start + i, "claim": f"{words[0]} {words[1]} {words[2]}",
                        "evidence": {str(did): [{"sentences": [0], "label": "SUPPORT"}]},
                        "cited_doc_ids": [did]})
        (root / f"claims_{prefix}.jsonl").write_text("\n".join(json.dumps(c) for c in out))

    claims("train", 0, 18)
    claims("dev", 100, 9)
    (root / "claims_test.jsonl").write_text(json.dumps({"id": 200, "claim": "heart"}))


def test_study_end_to_end_and_report(tmp_path, capsys):
    write_scifact(tmp_path)
    out = tmp_path / "res.json"
    res = run_study(tmp_path, out, seeds=(0, 1), pretrain_epochs=2, finetune_epochs=2)
    assert out.exists()
    s = res["systems"]
    assert {"bm25 (k1=1.2, b=0.75)", "bm25 tuned", "LSA", "neural (+ fine-tuned)",
            "RRF(bm25 tuned, LSA, neural)"} <= set(s)
    assert s["bm25 tuned"]["metrics"]["recall@100"]["mean"] == 1.0
    assert res["tuning"]["bm25"]["best"]["train_ndcg@10"] >= res["tuning"]["bm25"][
        "default_train_ndcg@10"] - 1e-9  # tuned on train can't lose to default on train
    assert "title only" in res["ablations"] and "errors" in res
    text = render(res)
    assert "| bm25 tuned |" in text and "Error analysis" in text
    assert main(["report", "--results", str(out)]) == 0
    assert "PaperMind retrieval results" in capsys.readouterr().out
