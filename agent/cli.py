"""Terminal entry: python -m agent \"find the clip where we're on the boat\"."""
from __future__ import annotations

import argparse

from framefinder.embedder import ClipEmbedder
from framefinder.store import DEFAULT_INDEX_DIR, Index

from .graph import run_request
from .llm import CursorCaller, load_env


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Search indexed videos with a plan, a visual check, and one retry loop.",
    )
    parser.add_argument("request", help="what to find, in a sentence")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--retries", type=int, default=2, help="how many times to rewrite the search (default 2)")
    args = parser.parse_args()

    index = Index(args.index_dir)
    if not index.manifest["videos"]:
        raise SystemExit("Index is empty. Run `python -m framefinder index <folder>` first.")
    caller = CursorCaller()
    caller.ensure_model()
    embedder = ClipEmbedder()
    index.check_model(embedder.name)
    result = run_request(args.request, caller, embedder, index, retries_left=args.retries)
    print(result["answer"])


if __name__ == "__main__":
    main()
