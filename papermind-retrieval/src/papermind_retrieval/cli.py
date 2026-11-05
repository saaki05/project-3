"""``papermind-retrieval study --data DIR --out results/scifact.json`` and ``report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papermind_retrieval.study import run_study


def render(result: dict) -> str:
    lines = [
        "# PaperMind retrieval results on SciFact dev (generated from the result JSON)", "",
        f"Corpus: {result['n_docs']} abstracts. Dev claims with evidence: "
        f"{result['n_dev_claims_with_evidence']}. Hyper-parameters tuned on "
        f"{result['n_train_claims_with_evidence']} training claims. Seeds: {result['seeds']}.", "",
        "| system | nDCG@10 [95% CI] | MRR@10 | Recall@10 | Recall@100 | Δ nDCG@10 vs default "
        "(paired bootstrap p) |", "|---|---|---|---|---|---|"]

    def row(name, s):
        m = s["metrics"]
        v = s.get("vs_default")
        delta = "baseline"
        if v:
            delta = f"{v['ndcg@10']['diff']:+.3f} (p={v['ndcg@10']['p_value']:.3f})"
        n = m["ndcg@10"]
        return (f"| {name} | {n['mean']:.3f} [{n['ci95'][0]:.3f}, {n['ci95'][1]:.3f}] | "
                f"{m['mrr@10']['mean']:.3f} | {m['recall@10']['mean']:.3f} | "
                f"{m['recall@100']['mean']:.3f} | {delta} |")

    for name, s in result["systems"].items():
        lines.append(row(name, s))
    lines += ["", "## Preprocessing and field ablations (default BM25 parameters)", "",
              "| variant | nDCG@10 [95% CI] | MRR@10 | Recall@10 | Recall@100 | "
              "Δ nDCG@10 vs default |",
              "|---|---|---|---|---|---|"]
    for name, s in result["ablations"].items():
        lines.append(row(name, s))
    t = result["tuning"]
    lines += ["", "## Tuning (on training claims)", "",
              f"- BM25: best k1={t['bm25']['best']['k1']}, b={t['bm25']['best']['b']} "
              f"(train nDCG@10 {t['bm25']['best']['train_ndcg@10']:.3f} vs default "
              f"{t['bm25']['default_train_ndcg@10']:.3f})",
              f"- RM3: {t['rm3']['best']}", f"- LSA: best dims {t['lsa']['best_dims']}", "",
              "## Error analysis", "", "```", json.dumps(
                  {k: v for k, v in result["errors"].items() if k != "examples_missed_at_100"},
                  indent=2), "```"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="papermind-retrieval")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("study")
    s.add_argument("--data", required=True, help="directory with corpus.jsonl and claims_*.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r = sub.add_parser("report")
    r.add_argument("--results", required=True)
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.command == "study":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res = run_study(a.data, a.out, tuple(a.seeds))
        print(render(res))
    else:
        text = render(json.loads(Path(a.results).read_text()))
        if a.out:
            Path(a.out).write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
