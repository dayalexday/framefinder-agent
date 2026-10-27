"""Upload eval/dataset.json to LangSmith and score plain CLIP search against the agent.

Each example looks like:
  {"question": "the boat on the ocean", "video": "trip.mp4", "timestamp_s": 74, "tolerance_s": 10}

A point counts when a returned clip's path contains `video` and its timestamp is within
`tolerance_s` of `timestamp_s`. The script does not invent questions or a score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framefinder.embedder import ClipEmbedder  # noqa: E402
from framefinder.search import search  # noqa: E402
from framefinder.store import DEFAULT_INDEX_DIR, Index  # noqa: E402

from agent.graph import run_request  # noqa: E402
from agent.llm import CursorCaller, load_env  # noqa: E402
from agent.nodes import LOOK_COUNT  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "dataset.json"
DATASET_NAME = "framefinder-agent"
REQUIRED = ("question", "video", "timestamp_s", "tolerance_s")


def load_examples() -> list[dict]:
    payload = json.loads(DATA_PATH.read_text())
    examples = payload.get("examples", [])
    if not isinstance(examples, list):
        raise SystemExit(f"{DATA_PATH} field 'examples' must be a list.")
    for i, example in enumerate(examples, 1):
        missing = [key for key in REQUIRED if key not in example]
        if missing:
            raise SystemExit(f"Example {i} is missing {', '.join(missing)}.")
    return examples


def is_hit(clips: list[dict], reference: dict) -> bool:
    needle = str(reference["video"])
    timestamp = float(reference["timestamp_s"])
    tolerance = float(reference["tolerance_s"])
    for clip in clips:
        if needle in str(clip["video"]) and abs(float(clip["t"]) - timestamp) <= tolerance:
            return True
    return False


def grade(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "hit", "score": 1 if is_hit(outputs.get("clips") or [], reference_outputs) else 0}


def plain_clips(question: str, embedder, index) -> dict:
    hits = search(embedder, index, question, top_k=5, min_gap_s=6.0)
    return {"clips": [{"video": video, "t": t, "score": score} for video, t, score in hits]}


def agent_clips(question: str, caller, embedder, index) -> dict:
    state = run_request(question, caller, embedder, index)
    rows = state["hits"][:LOOK_COUNT]
    if state["verdict"] and state.get("matched_indices"):
        rows = [rows[i] for i in state["matched_indices"] if i < len(rows)]
    return {
        "clips": [
            {"video": hit["video"], "t": hit["t"], "score": hit["score"], "caption": hit["caption"]}
            for hit in rows
        ]
    }


def sync_dataset(client, examples: list[dict]):
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(DATASET_NAME, description="FrameFinder plain search vs agent")
    for existing in client.list_examples(dataset_id=dataset.id):
        client.delete_example(existing.id)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": example["question"]},
                "reference_outputs": {
                    "video": example["video"],
                    "timestamp_s": example["timestamp_s"],
                    "tolerance_s": example["tolerance_s"],
                },
            }
            for example in examples
        ],
    )
    return dataset


def main() -> None:
    load_env()
    examples = load_examples()
    if not examples:
        raise SystemExit(
            f"{DATA_PATH} has no examples, so there is nothing to score. "
            'Add about 20 questions you can answer from the index, each like '
            '{"question": "...", "video": "clip.mp4", "timestamp_s": 74, "tolerance_s": 10}.'
        )
    import os

    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        raise SystemExit("Set LANGSMITH_API_KEY in .env before uploading a scorecard.")

    index = Index(DEFAULT_INDEX_DIR)
    if not index.manifest["videos"]:
        raise SystemExit("Index is empty. Run `python -m framefinder index <folder>` first.")
    caller = CursorCaller()
    caller.ensure_model()
    embedder = ClipEmbedder()
    index.check_model(embedder.name)

    from langsmith import Client, evaluate

    client = Client()
    sync_dataset(client, examples)

    def plain_target(inputs: dict) -> dict:
        return plain_clips(inputs["question"], embedder, index)

    def agent_target(inputs: dict) -> dict:
        return agent_clips(inputs["question"], caller, embedder, index)

    print("Scoring plain search...")
    evaluate(
        plain_target,
        data=DATASET_NAME,
        evaluators=[grade],
        experiment_prefix="plain-search",
        max_concurrency=1,
    )
    print("Scoring agent...")
    evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[grade],
        experiment_prefix="agent",
        max_concurrency=1,
    )
    print(f"Done. Compare the two experiments in LangSmith project {os.environ.get('LANGSMITH_PROJECT', DATASET_NAME)}.")


if __name__ == "__main__":
    main()
