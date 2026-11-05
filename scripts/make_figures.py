"""Generate every PaperMind figure from papermind-retrieval/results/scifact.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = json.load(open(ROOT / "papermind-retrieval" / "results" / "scifact.json"))
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fs.setup()


def colour(name):
    if name.startswith("bm25 (k1"):
        return fs.INK
    if name.startswith("bm25 tuned"):
        return fs.BLUE
    if "neural" in name:
        return fs.ORANGE
    if "LSA" in name or "min-max" in name or "RRF" in name:
        return fs.VIOLET
    return fs.GRAY


def systems():
    rows = [(n, s["metrics"]["ndcg@10"]) for n, s in D["systems"].items()]
    rows.sort(key=lambda r: r[1]["mean"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (n, m) in enumerate(rows):
        ax.plot(m["ci95"], [i, i], color=colour(n), lw=2.6)
        ax.scatter([m["mean"]], [i], color=colour(n), s=48, zorder=3, edgecolor=fs.SURFACE)
        ax.text(m["ci95"][1] + 0.012, i, f"{m['mean']:.3f}", va="center", fontsize=8.5,
                color=fs.INK2)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows], fontsize=8.8)
    ax.set_xlim(0.2, 0.97)
    ax.set_xlabel("nDCG@10 on SciFact dev claims (dot = mean, bar = 95% bootstrap interval)")
    ax.set_title("Retrieval quality of every system on SciFact dev claims")
    ax.grid(axis="y", visible=False)
    base = D["systems"]["bm25 (k1=1.2, b=0.75)"]["metrics"]["ndcg@10"]["mean"]
    ax.axvline(base, color=fs.INK, ls=":", lw=1.2)
    fs.save(fig, OUT / "systems_ndcg.png")


def deltas():
    rows = []
    for n, s in D["systems"].items():
        if "vs_default" in s:
            rows.append((n, s["vs_default"]["ndcg@10"]))
    for n, s in D["ablations"].items():
        rows.append((f"BM25 {n}", s["vs_default"]["ndcg@10"]))
    rows.sort(key=lambda r: r[1]["diff"])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axvline(0, color=fs.MUTED, lw=1)
    for i, (n, c) in enumerate(rows):
        sig = c["p_value"] < 0.05
        col = fs.ORANGE if sig and c["diff"] < 0 else (fs.BLUE if sig else fs.GRAY)
        ax.plot([c["ci_low"], c["ci_high"]], [i, i], color=col, lw=2.4)
        ax.scatter([c["diff"]], [i], color=col, s=42, zorder=3, edgecolor=fs.SURFACE)
        ax.text(1.02, i, f"{c['diff']:+.3f}   p={c['p_value']:.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.3, color=fs.INK2)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows], fontsize=8.6)
    ax.set_xlabel("change in nDCG@10 vs default BM25 (paired bootstrap 95% interval)")
    ax.set_title("Change in nDCG@10 relative to default BM25")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=fs.ORANGE, label="significantly worse (p < 0.05)")
    ax.scatter([], [], color=fs.GRAY, label="not significant")
    ax.legend(loc="lower right")
    fs.save(fig, OUT / "delta_vs_default.png")


def bm25_grid():
    grid = D["tuning"]["bm25"]["grid"]
    k1s = sorted({float(k.split(",")[0]) for k in grid})
    bs = sorted({float(k.split(",")[1]) for k in grid})
    z = np.array([[grid[f"{k},{b}"] for b in bs] for k in k1s])
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.imshow(z, cmap="Blues", aspect="auto", vmin=z.min() - 0.005)
    for i in range(len(k1s)):
        for j in range(len(bs)):
            ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if (z[i, j] - z.min()) / (z.max() - z.min()) > 0.4 else fs.INK)
    ax.set_xticks(range(len(bs)), [str(b) for b in bs])
    ax.set_yticks(range(len(k1s)), [str(k) for k in k1s])
    ax.set_xlabel("b (length normalisation)")
    ax.set_ylabel("k1 (term-frequency saturation)")
    ax.set_title("BM25 is flat: train nDCG@10 spans only %.3f" % (z.max() - z.min()))
    ax.grid(False)
    fs.save(fig, OUT / "bm25_grid.png")


def recall():
    rows = [(n, s["metrics"]["recall@100"]["mean"], s["metrics"]["ndcg@10"]["mean"])
            for n, s in D["systems"].items()]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for n, r, nd in rows:
        ax.scatter([nd], [r], color=colour(n), s=60, edgecolor=fs.SURFACE, zorder=3)
    offsets = {"bm25 (k1=1.2, b=0.75)": ("BM25 (default)", (8, -14)),
               "min-max(bm25 tuned, LSA)": ("min-max fusion", (-20, 22)),
               "RRF(bm25 tuned, LSA, neural)": ("RRF with LSA + neural", (-70, 26)),
               "LSA": ("LSA", (6, 8)), "neural (+ fine-tuned)": ("neural, fine-tuned", (8, -3)),
               "neural (pretrained only)": ("neural, pretrained only", (8, -3))}
    for n, r, nd in rows:
        if n in offsets:
            lab, off = offsets[n]
            ax.annotate(lab, (nd, r), xytext=off, textcoords="offset points", fontsize=8.2,
                        color=fs.INK2, arrowprops={"arrowstyle": "-", "color": fs.MUTED, "lw": 0.8})
    ax.set_xlabel("nDCG@10 (ranking quality at the top)")
    ax.set_ylabel("Recall@100 (coverage of relevant abstracts)")
    ax.set_title("Top-rank quality against recall at 100")
    fs.save(fig, OUT / "ndcg_vs_recall.png")


def architecture():
    fs.pipeline(OUT / "architecture.png", [
        ("ingestion", ["SciFact loader + validation", "tokenise, stop-words, Porter", "checksummed snapshot"]),
        ("retrieval", ["BM25 (from scratch)", "RM3 feedback · LSA", "hashed neural encoder",
                       "RRF / min-max fusion", "bootstrap + permutation tests"]),
        ("platform", ["/search → ranked abstracts", "best evidence sentence", "/documents/{id}",
                      "FastAPI service"]),
    ], "PaperMind: three packages, one-way dependencies", accent=fs.BLUE, width=10.5)


if __name__ == "__main__":
    systems()
    deltas()
    bm25_grid()
    recall()
    architecture()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
