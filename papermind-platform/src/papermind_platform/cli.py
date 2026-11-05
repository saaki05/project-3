"""``papermind-serve --data DIR`` where DIR holds a ``papermind-ingest build`` snapshot."""

from __future__ import annotations

import argparse

import uvicorn
from papermind_ingest import read_corpus

from papermind_platform.api import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="papermind-serve")
    ap.add_argument("--data", required=True, help="snapshot directory (corpus.jsonl + manifest)")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(read_corpus(a.data), a.k1, a.b), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
