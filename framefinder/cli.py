from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .frames import find_videos, sample_frames
from .store import DEFAULT_INDEX_DIR, Index

EMBED_CHUNK = 64  # frames held in memory at once while indexing


def cmd_index(args) -> None:
    from .embedder import ClipEmbedder

    videos = find_videos(args.folder)
    if not videos:
        raise SystemExit(f"No videos found under {args.folder}")

    embedder = ClipEmbedder()
    index = Index(args.index_dir)
    if args.force:
        index.manifest = {"model": None, "videos": {}}
    index.check_model(embedder.name)
    removed = index.prune_missing()
    if removed:
        print(f"Removed {removed} missing video(s) from the index.")

    for n, video in enumerate(videos, 1):
        label = f"[{n}/{len(videos)}] {video.name}"
        if not args.force and index.is_current(video, args.every):
            print(f"{label}: up to date, skipping")
            continue
        start = time.time()
        times, chunks, pending, pending_t = [], [], [], []
        try:
            for t, img in sample_frames(video, args.every):
                pending.append(img)
                pending_t.append(t)
                if len(pending) >= EMBED_CHUNK:
                    chunks.append(embedder.embed_images(pending))
                    times += pending_t
                    pending, pending_t = [], []
                    print(f"{label}: {len(times)} frames...", end="\r", flush=True)
            if pending:
                chunks.append(embedder.embed_images(pending))
                times += pending_t
        except RuntimeError as e:
            print(f"{label}: skipped ({e})")
            continue
        if not times:
            print(f"{label}: no frames read, skipped")
            continue
        import numpy as np

        index.add(video, times, np.concatenate(chunks, axis=0), args.every)
        print(f"{label}: {len(times)} frames in {time.time() - start:.0f}s        ")
    print("Index ready.")


def cmd_search(args) -> None:
    from .embedder import ClipEmbedder
    from .search import fmt_time, save_results, search

    index = Index(args.index_dir)
    if not index.manifest["videos"]:
        raise SystemExit("Index is empty. Run `index <folder>` first.")
    embedder = ClipEmbedder()
    index.check_model(embedder.name)

    hits = search(embedder, index, args.query, args.top, args.gap)
    for i, (video, t, score) in enumerate(hits, 1):
        print(f"{i:2d}. {score:.3f}  {fmt_time(t)}  {video}")

    if hits and not args.no_thumbs:
        out = save_results(hits, args.query, args.results_dir)
        print(f"\nThumbnails + contact sheet: {out}/results.html")
        if args.open and sys.platform == "darwin":
            subprocess.run(["open", str(out / "results.html")], check=False)


def main() -> None:
    p = argparse.ArgumentParser(prog="framefinder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="embed frames from every video in a folder")
    pi.add_argument("folder")
    pi.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames (default 2)")
    pi.add_argument("--force", action="store_true", help="rebuild everything")
    pi.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="find frames matching a text description")
    ps.add_argument("query")
    ps.add_argument("-n", "--top", type=int, default=10)
    ps.add_argument("--gap", type=float, default=6.0, help="min seconds between hits in the same video")
    ps.add_argument("--open", action="store_true", help="open the contact sheet in your browser (macOS)")
    ps.add_argument("--no-thumbs", action="store_true")
    ps.add_argument("--results-dir", default="results")
    ps.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    ps.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)
