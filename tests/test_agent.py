"""The retry loop, without Cursor or CLIP. Run: python tests/test_agent.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import run_request  # noqa: E402
from agent.llm import CursorCaller, extract_json  # noqa: E402
from agent.nodes import format_answer  # noqa: E402
from agent.state import SearchPhrase  # noqa: E402
from framefinder.search import search  # noqa: E402
from framefinder.store import Index  # noqa: E402

COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # BGR


def make_video(path: Path, seconds=9, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert w.isOpened(), "VideoWriter could not open (codec missing?)"
    names = list(COLORS)
    for i in range(seconds * fps):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:] = COLORS[names[i // (fps * 3)]]
        w.write(frame)
    w.release()


class FakeEmbedder:
    name = "fake"

    def embed_images(self, images):
        v = np.array([np.asarray(im).reshape(-1, 3).mean(0) for im in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_text(self, text):
        rgb = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}[text]
        return np.array(rgb, dtype=np.float32)


class ScriptedCaller:
    """Returns canned JSON shapes in the order the graph asks for them."""

    def __init__(self, phrases: list[str], verdicts: list[tuple[bool, str, list[int] | None]]):
        self.phrases = list(phrases)
        self.verdicts = list(verdicts)
        self.calls: list[str] = []

    def complete(self, prompt, schema, images=None):
        self.calls.append(schema.__name__)
        if schema.__name__ == "SearchPhrase":
            return schema(phrase=self.phrases.pop(0))
        if schema.__name__ == "Verdict":
            match, reason, frames = self.verdicts.pop(0)
            if frames is None:
                frames = [1] if match else []
            return schema(match=match, reason=reason, matched_frames=frames)
        n = len(images or [])
        return schema(captions=[f"caption {i}" for i in range(n)])


def indexed_video():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    video = root / "clip.mp4"
    make_video(video)
    from framefinder.frames import sample_frames

    samples = list(sample_frames(video, every_s=1.0))
    emb = FakeEmbedder()
    index = Index(root / "idx")
    index.check_model(emb.name)
    index.add(video, [t for t, _ in samples], emb.embed_images([im for _, im in samples]), 1.0)
    return tmp, emb, Index(root / "idx")


def test_yes_stops():
    tmp, emb, index = indexed_video()
    try:
        caller = ScriptedCaller(["green"], [(True, "the frame is green", [1])])
        result = run_request("show me green", caller, emb, index, retries_left=2)
        assert caller.calls == ["SearchPhrase", "FrameNotes", "Verdict"], caller.calls
        assert [a["phrase"] for a in result["attempts"]] == ["green"]
        assert result["verdict"] is True
        assert result["matched_indices"] == [0]
        assert 3 <= result["hits"][0]["t"] < 6
        assert "couldn't find a match" not in result["answer"]
        assert "Match" in result["answer"]
        assert "caption 0" in result["answer"]
        assert "near misses:" not in result["answer"]
    finally:
        tmp.cleanup()


def test_no_then_better_search():
    tmp, emb, index = indexed_video()
    try:
        caller = ScriptedCaller(
            ["red", "green"],
            [(False, "that is not the boat", []), (True, "green matches", [1])],
        )
        result = run_request("the boat", caller, emb, index, retries_left=2)
        assert [a["phrase"] for a in result["attempts"]] == ["red", "green"]
        assert caller.calls == [
            "SearchPhrase",
            "FrameNotes",
            "Verdict",
            "SearchPhrase",
            "FrameNotes",
            "Verdict",
        ], caller.calls
        assert result["verdict"] is True
        assert 3 <= result["hits"][0]["t"] < 6, result["hits"][0]
        assert result["retries_left"] == 1
    finally:
        tmp.cleanup()


def test_retries_exhausted_admits_miss():
    tmp, emb, index = indexed_video()
    try:
        caller = ScriptedCaller(
            ["red", "green", "blue"],
            [(False, "no", []), (False, "still no", []), (False, "nothing matches", [])],
        )
        result = run_request("a boat on the ocean", caller, emb, index, retries_left=2)
        assert [a["phrase"] for a in result["attempts"]] == ["red", "green", "blue"]
        assert caller.calls.count("SearchPhrase") == 3
        assert caller.calls.count("Verdict") == 3
        assert result["verdict"] is False
        assert result["retries_left"] == 0
        assert result["answer"].startswith("couldn't find a match")
        assert "closest:" in result["answer"]
        assert 6 <= result["hits"][0]["t"] < 9
    finally:
        tmp.cleanup()


def test_format_answer_labels_closest():
    text = format_answer(
        {
            "request": "boat",
            "search_phrase": "boat",
            "hits": [{"video": "/tmp/a.mp4", "t": 4.2, "score": 0.2, "caption": "a red square"}],
            "verdict": False,
            "reason": "no boat",
            "matched_indices": [],
            "retries_left": 0,
            "attempts": [],
            "answer": "",
        }
    )
    assert text.startswith("couldn't find a match")
    assert "closest:" in text
    assert "00:04" in text
    assert "a red square" in text


def test_format_answer_splits_near_misses():
    text = format_answer(
        {
            "request": "boat",
            "search_phrase": "boat",
            "hits": [
                {"video": "/tmp/boat.mp4", "t": 32.0, "score": 0.3, "caption": "person on a boat deck"},
                {"video": "/tmp/coast.mp4", "t": 4.0, "score": 0.2, "caption": "rocky coastline"},
                {"video": "/tmp/beach.mp4", "t": 34.0, "score": 0.2, "caption": "kids on a beach"},
            ],
            "verdict": True,
            "reason": "frame 1 is on a boat",
            "matched_indices": [0],
            "retries_left": 2,
            "attempts": [],
            "answer": "",
        }
    )
    assert text.startswith("Match")
    assert "person on a boat deck" in text
    assert "near misses:" in text
    assert text.index("person on a boat deck") < text.index("near misses:")
    assert text.index("near misses:") < text.index("rocky coastline")
    assert "kids on a beach" in text


def test_extract_json_and_one_retry():
    assert json.loads(extract_json('sure\n```json\n{"phrase": "boat"}\n```'))["phrase"] == "boat"
    replies = iter(["nope", '{"phrase": "boat on the ocean"}'])
    caller = CursorCaller(transport=lambda prompt, images: next(replies))
    got = caller.complete("turn this into a phrase", SearchPhrase)
    assert got.phrase == "boat on the ocean"

    caller = CursorCaller(transport=lambda prompt, images: "still not json")
    try:
        caller.complete("turn this into a phrase", SearchPhrase)
    except RuntimeError as exc:
        assert "still not json" in str(exc)
    else:
        raise AssertionError("expected the second bad reply to fail")


def test_eval_refuses_to_score_without_questions():
    from eval.run_eval import grade, is_hit, load_examples, main

    assert load_examples() == []
    reference = {"video": "trip.mp4", "timestamp_s": 74, "tolerance_s": 10}
    assert is_hit([{"video": "/films/trip.mp4", "t": 80}], reference)
    assert not is_hit([{"video": "/films/trip.mp4", "t": 100}], reference)
    assert not is_hit([{"video": "/films/other.mp4", "t": 74}], reference)
    assert grade({"clips": [{"video": "trip.mp4", "t": 74}]}, reference)["score"] == 1
    try:
        main()
    except SystemExit as exc:
        assert "nothing to score" in str(exc.code)
    else:
        raise AssertionError("empty dataset should refuse to score")


def test_search_still_ranks_colors():
    """The agent tests above depend on the same local ranker the CLI uses."""
    tmp, emb, index = indexed_video()
    try:
        hits = search(emb, index, "blue", top_k=1, min_gap_s=1.0)
        assert hits and hits[0][1] >= 6
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    test_yes_stops()
    test_no_then_better_search()
    test_retries_exhausted_admits_miss()
    test_format_answer_labels_closest()
    test_format_answer_splits_near_misses()
    test_extract_json_and_one_retry()
    test_eval_refuses_to_score_without_questions()
    test_search_still_ranks_colors()
    print("ok")
