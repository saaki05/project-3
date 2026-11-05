"""``papermind-ingest build --raw DIR --out DIR`` and ``papermind-ingest stats --raw DIR``."""

from __future__ import annotations

import argparse

from papermind_ingest.scifact import load_scifact
from papermind_ingest.store import write_corpus


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="papermind-ingest")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="validate SciFact files and write a normalised snapshot")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("stats", help="print benchmark statistics")
    s.add_argument("--raw", required=True)
    args = ap.parse_args(argv)

    bench = load_scifact(args.raw)
    if args.command == "build":
        m = write_corpus(bench.corpus, args.out)
        print(f"wrote {m['num_documents']} documents to {args.out}")
        return 0
    for name in ("train", "dev", "test"):
        claims = getattr(bench, name)
        with_ev = sum(c.has_evidence for c in claims)
        print(f"{name}: {len(claims)} claims, {with_ev} with evidence")
    print(f"corpus: {len(bench.corpus)} abstracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
