"""Tests for everything except the MLX model itself (uses a fake embedder).

Run:  python -m pytest tests -q     (or: python tests/test_logic.py)
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framefinder.frames import find_videos, sample_frames  # noqa: E402
from framefinder.search import rank, save_results, search  # noqa: E402
from framefinder.store import Index  # noqa: E402

COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # BGR


def make_video(path: Path, seconds=9, fps=10):
    """3 s red, 3 s green, 3 s blue."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert w.isOpened(), "VideoWriter could not open (codec missing?)"
    names = list(COLORS)
    for i in range(seconds * fps):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:] = COLORS[names[i // (fps * 3)]]
        w.write(frame)
    w.release()


class FakeEmbedder:
    """Embeds an image as its mean colour (R,G,B) and text as the named colour."""

    name = "fake"

    def embed_images(self, images):
        v = np.array([np.asarray(im).reshape(-1, 3).mean(0) for im in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_text(self, text):
        rgb = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}[text]
        return np.array(rgb, dtype=np.float32)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "clips").mkdir()
        make_video(d / "clips" / "a.mp4")
        assert [p.name for p in find_videos(d)] == ["a.mp4"]

        samples = list(sample_frames(d / "clips" / "a.mp4", every_s=1.0))
        assert 8 <= len(samples) <= 9
        assert samples[0][0] == 0.0

        emb = FakeEmbedder()
        index = Index(d / "idx")
        index.check_model(emb.name)
        times = [t for t, _ in samples]
        index.add(d / "clips" / "a.mp4", times, emb.embed_images([im for _, im in samples]), 1.0)

        # Reload from disk and check incremental bookkeeping.
        index = Index(d / "idx")
        assert index.is_current(d / "clips" / "a.mp4", 1.0)
        assert not index.is_current(d / "clips" / "a.mp4", 2.0)

        hits = search(emb, index, "green", top_k=3, min_gap_s=1.5)
        assert hits and 3 <= hits[0][1] < 6, hits  # green section is 3-6 s
        ts = sorted(h[1] for h in hits)
        assert all(b - a >= 1.5 for a, b in zip(ts, ts[1:])), "near-duplicates not suppressed"

        out = save_results(hits, "green", d / "results")
        assert (out / "results.html").exists() and list(out.glob("*.jpg"))


def test_rank_empty():
    assert rank(np.ones(3, np.float32), np.zeros((0, 1), np.float32), [], 5, 1.0) == []


if __name__ == "__main__":
    test_end_to_end()
    test_rank_empty()
    print("ok")
