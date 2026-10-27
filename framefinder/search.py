"""Ranking and result rendering."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .frames import grab_frame
from .store import Index

Hit = Tuple[str, float, float]  # (video path, timestamp seconds, cosine score)


def fmt_time(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def rank(query_emb: np.ndarray, embs: np.ndarray, meta: list, top_k: int, min_gap_s: float) -> List[Hit]:
    """Top-k frames by cosine similarity, skipping near-duplicates (same video, within min_gap_s)."""
    if len(meta) == 0:
        return []
    scores = embs @ query_emb
    picked: List[Hit] = []
    for i in np.argsort(-scores):
        video, t = meta[i]
        if any(video == pv and abs(t - pt) < min_gap_s for pv, pt, _ in picked):
            continue
        picked.append((video, t, float(scores[i])))
        if len(picked) >= top_k:
            break
    return picked


def search(embedder, index: Index, query: str, top_k: int = 10, min_gap_s: float = 6.0) -> List[Hit]:
    embs, meta = index.load_all()
    return rank(embedder.embed_text(query), embs, meta, top_k, min_gap_s)


def save_results(hits: List[Hit], query: str, out_root: str | Path = "results") -> Path:
    """Write thumbnails and a results.html contact sheet; return the output folder."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:50] or "query"
    out = Path(out_root) / slug
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    cards = []
    for rank_i, (video, t, score) in enumerate(hits, 1):
        img = grab_frame(video, t)
        if img is None:
            continue
        img.thumbnail((640, 640))
        name = f"{rank_i:02d}_{Path(video).stem[:30]}_{fmt_time(t).replace(':', '-')}.jpg"
        img.save(out / name, quality=88)
        link = f"file://{Path(video).resolve()}#t={int(t)}"
        cards.append(
            f'<figure><img src="{html.escape(name)}"><figcaption>'
            f"<b>#{rank_i}</b> {html.escape(Path(video).name)} @ {fmt_time(t)} "
            f'(score {score:.3f}) · <a href="{html.escape(link)}">open</a></figcaption></figure>'
        )
    page = (
        "<!doctype html><meta charset=utf-8><title>Results</title>"
        "<style>body{font:14px -apple-system,sans-serif;margin:24px;background:#111;color:#eee}"
        "a{color:#8ab4ff}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}"
        "img{width:100%;border-radius:6px}figure{margin:0}</style>"
        f"<h2>{html.escape(query)}</h2><div class=g>{''.join(cards)}</div>"
    )
    (out / "results.html").write_text(page)
    return out
