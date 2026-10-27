"""One function per box in the flowchart. Search is local CLIP. The other boxes ask Composer."""
from __future__ import annotations

import io
import sys

from framefinder.frames import grab_frame
from framefinder.search import fmt_time, search

from .llm import ModelCaller
from .state import AgentState, Hit, SearchPhrase, Verdict, frame_notes_model

LOOK_COUNT = 3
MAX_SIDE = 512


def _say(line: str) -> None:
    print(line, file=sys.stderr)


def frame_jpeg(video: str, t: float) -> bytes | None:
    img = grab_frame(video, t)
    if img is None:
        return None
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def format_answer(state: AgentState) -> str:
    rows = state["hits"][:LOOK_COUNT]
    lines: list[str] = []
    if state["verdict"]:
        lines.append("Match")
    else:
        lines.append("couldn't find a match")
        if state.get("reason"):
            lines.append(state["reason"])
        lines.append("closest:")
    if not rows:
        lines.append("(no frames)")
    for i, hit in enumerate(rows, 1):
        lines.append(f"{i}. {fmt_time(hit['t'])}  {hit['video']}")
        lines.append(f"   {hit.get('caption') or '(no caption)'}")
    return "\n".join(lines)


def make_nodes(caller: ModelCaller, embedder, index, *, top_k: int = 5, min_gap_s: float = 6.0):
    def plan(state: AgentState) -> dict:
        result = caller.complete(
            "Turn this video request into a short search phrase for a CLIP model. "
            "CLIP compares the phrase to still frames. Keep objects, places, colors, and visible actions. "
            "Drop dialogue, plot, and anything a single frame cannot show.\n"
            f"Request: {state['request']}",
            SearchPhrase,
        )
        phrase = result.phrase.strip()
        _say(f"plan: {phrase}")
        return {"search_phrase": phrase}

    def search_node(state: AgentState) -> dict:
        found = search(embedder, index, state["search_phrase"], top_k=top_k, min_gap_s=min_gap_s)
        hits: list[Hit] = [
            {"video": video, "t": float(t), "score": float(score), "caption": ""}
            for video, t, score in found
        ]
        _say(f"search: {state['search_phrase']!r} -> {len(hits)} hit(s)")
        return {"hits": hits}

    def look(state: AgentState) -> dict:
        chosen = state["hits"][:LOOK_COUNT]
        images: list[bytes] = []
        indexes: list[int] = []
        for i, hit in enumerate(chosen):
            blob = frame_jpeg(hit["video"], hit["t"])
            if blob is None:
                continue
            images.append(blob)
            indexes.append(i)
        captions = [""] * len(chosen)
        if images:
            notes = caller.complete(
                "Describe each attached frame in one sentence about what is visible. "
                "The first image is frame 1, then frame 2, in order. "
                f"Return exactly {len(images)} captions. Do not guess at frames you were not shown.",
                frame_notes_model(len(images)),
                images,
            )
            for index, caption in zip(indexes, notes.captions):
                captions[index] = caption.strip()
        updated: list[Hit] = []
        for i, hit in enumerate(state["hits"]):
            caption = captions[i] if i < len(captions) else ""
            updated.append({**hit, "caption": caption})
        for i, caption in enumerate(captions, 1):
            if caption:
                _say(f"look {i}: {caption}")
        return {"hits": updated}

    def check(state: AgentState) -> dict:
        if not state["hits"]:
            verdict = Verdict(match=False, reason="search returned no frames")
        else:
            lines = []
            for i, hit in enumerate(state["hits"][:LOOK_COUNT], 1):
                lines.append(f"{i}. {hit['caption'] or '(frame unreadable)'}")
            verdict = caller.complete(
                "Decide whether the frames match the request. A weak resemblance is not a match. "
                "Match only if at least one caption shows what was asked for.\n"
                f"Request: {state['request']}\n"
                f"Search phrase: {state['search_phrase']}\n"
                "Captions:\n" + "\n".join(lines),
                Verdict,
            )
        attempt = {
            "phrase": state["search_phrase"],
            "captions": [h["caption"] for h in state["hits"][:LOOK_COUNT]],
            "match": verdict.match,
            "reason": verdict.reason,
        }
        _say(f"check: {'yes' if verdict.match else 'no'} ({verdict.reason})")
        return {
            "verdict": verdict.match,
            "reason": verdict.reason,
            "attempts": [*state["attempts"], attempt],
        }

    def refine(state: AgentState) -> dict:
        used = [a["phrase"] for a in state["attempts"]]
        history = []
        for attempt in state["attempts"]:
            caps = "; ".join(c for c in attempt["captions"] if c) or "(no captions)"
            history.append(f"- {attempt['phrase']}: {caps}. Rejected because: {attempt['reason']}")
        prompt = (
            "The frames from the last search did not match. Write a new short visual CLIP phrase. "
            "Keep it to what a still frame can show, and do not repeat a phrase already tried.\n"
            f"Request: {state['request']}\n"
            "Already tried:\n" + "\n".join(history)
        )
        phrase = caller.complete(prompt, SearchPhrase).phrase.strip()
        if phrase.lower() in {u.lower() for u in used}:
            phrase = caller.complete(
                prompt + "\nDo not use any of these phrases: " + "; ".join(used),
                SearchPhrase,
            ).phrase.strip()
        left = state["retries_left"] - 1
        _say(f"refine: {phrase} ({left} retries left)")
        return {"search_phrase": phrase, "retries_left": left}

    def answer(state: AgentState) -> dict:
        text = format_answer(state)
        return {"answer": text}

    return plan, search_node, look, check, refine, answer


def route(state: AgentState) -> str:
    if state["verdict"] or state["retries_left"] <= 0:
        return "answer"
    return "refine"
