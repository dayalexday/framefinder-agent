# FrameFinder Agent

I already built FrameFinder: you type a short description, and CLIP finds matching frames in a folder of videos. It runs fully offline on an Apple Silicon Mac (CLIP on [MLX](https://github.com/ml-explore/mlx)).

This project puts a small agent on top of it. Instead of matching only the exact words you type, you can ask something like *"find the clip where we're on the boat."* The agent turns that into a search, runs FrameFinder, looks at the frames it got back, and decides whether they actually match. If they don't, it searches again with a better phrase. If it still can't find it, it says so instead of returning false positives.

## Example run

```
$ python -m agent "find the clip where we're on the boat"
plan: people on a boat on the water
search: 'people on a boat on the water' -> 5 hit(s)
look 1: A person with long blonde hair is seen from behind on a boat deck at night under red stage lighting.
look 2: A wide view shows a rocky coastline, deep blue water, and a small island under a clear blue sky.
look 3: Children play on a beach in golden backlight, with one child about to head a ball in silhouette against the bright sky.
check: yes (Caption 1 shows a person on a boat deck at night. Captions 2 and 3 are coastline and beach, not on a boat.)
Match
1. 00:32  snaptik_7659770540558650642_v3.mp4
   A person with long blonde hair is seen from behind on a boat deck at night under red stage lighting.
```

This one matched on the first search, so it never had to refine.

## How one request works

```
you type something
      │
      ▼
   plan     model writes a short visual search phrase
      │
      ▼
   search   local CLIP ranks the indexed frames (no cloud)
      │
      ▼
   look     model describes the top 3 frames, one sentence each
      │
      ▼
   check    model decides: do these match the original ask? yes/no + reason
      │
      ├─ yes, or out of retries → answer (the clips, or an honest miss)
      └─ no, with retries left  → refine (new phrase) → back to search
```

The default is 2 retries, so at most 3 searches. The answer step is plain Python with no model, so a miss really is a miss.

## Stack

- **CLIP on MLX** is still the search engine. It runs locally and it only matches text to images.
- **A chat/vision model** handles the three steps that need language or eyes: plan, look, and check (plus refine). It calls an AI API, currently Cursor's Composer model through the Cursor SDK, with tools turned off so it can only return text.
- **LangGraph** runs the flowchart. Each box is a normal Python function in `agent/nodes.py`, and `agent/graph.py` wires them together, including the check → refine → search loop. LangGraph carries the shared state (request, phrases tried, hits, captions, retries left) between steps.
- **LangSmith** records and grades runs. With tracing on, every run shows up there step by step, including exactly what the model was sent and what it said. `eval/run_eval.py` scores plain CLIP search against the agent on a set of questions with known answers.

CLIP stays local because the index already exists and sending every frame to a cloud model would be slow and expensive. The model only sees the few frames that matter.

## Setup

Apple Silicon, Python 3.10+ (3.12 works).

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt -r requirements-agent.txt
./setup.sh   # downloads CLIP into mlx_model/ (gitignored)
cp .env.example .env
```

Index a folder of videos once:

```bash
python -m framefinder index ~/Movies/my_videos
```

## Run it

```bash
python -m agent "find the clip where we're on the boat"
python -m agent "..." --retries 3
```

Each step prints as it runs. You get either a match with captions, or `couldn't find a match` plus the closest frames.

Plain FrameFinder still works on its own, and it's the baseline in the eval:

```bash
python -m framefinder search "people on a boat" --open
```

## Scorecard

Put questions you know the answer to in `eval/dataset.json`:

```json
{"question": "the boat on the ocean", "video": "trip.mp4", "timestamp_s": 74, "tolerance_s": 10}
```

Then run `python eval/run_eval.py`. Both plain search and the agent get the same grader: a hit counts when the right video shows up within `tolerance_s` of the timestamp. The two runs show up side by side in LangSmith. If the dataset is empty, it won't score anything.

## Layout

```
framefinder/     local CLIP index + search
agent/           LangGraph loop (graph.py, nodes.py, state.py, llm.py, cli.py)
eval/            LangSmith dataset + plain-vs-agent scorecard
tests/           offline tests with a fake model, no network
vendor/mlx_clip/ Apple's MLX CLIP reference code
```

Kept off GitHub: `.framefinder_index/` (it stores absolute paths to your files), `results/`, `mlx_model/`, `.env`.

<details>
<summary>Index and search flags</summary>

| Flag | Command | Meaning |
|---|---|---|
| `--every S` | index | Seconds between sampled frames (default 2) |
| `--force` | index | Rebuild from scratch |
| `-n N` | search | How many results (default 10) |
| `--gap S` | search | Min seconds between hits in the same video (default 6) |
| `--open` | search | Open the contact sheet |
| `--index-dir`, `--results-dir` | both | Override paths |

Scores around 0.2 to 0.35 are normal; the ranking matters more than the number. About an hour of video at the default sampling is around 1,800 frames and takes a few minutes to index on an M2. For a bigger CLIP model, run `./setup.sh openai/clip-vit-large-patch14`.
</details>
