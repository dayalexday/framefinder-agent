# framefinder

Describe a scene in plain English and find the matching moments in a folder of videos.
Runs fully offline on an Apple Silicon Mac using a CLIP model on [MLX](https://github.com/ml-explore/mlx).

## How it works

1. **Index**: samples a frame every N seconds from each video and turns it into an embedding with CLIP's image encoder.
2. **Search**: turns your description into an embedding with CLIP's text encoder and ranks frames by cosine similarity.
3. Results are printed with timestamps, and thumbnails plus a `results.html` contact sheet are saved.

Embeddings are cached in `.framefinder_index/`, so re-indexing only processes new or changed videos.

## Setup (once, needs internet)

```bash
cd framefinder
./setup.sh
```

This creates a `.venv`, installs dependencies, fetches Apple's MLX CLIP reference code
into `vendor/`, and downloads and converts the model weights into `mlx_model/`.
After that, no network is needed.

Want better accuracy at the cost of slower indexing? `./setup.sh openai/clip-vit-large-patch14`

## Use

```bash
source .venv/bin/activate

# Index a folder (recursive). Default: one frame every 2 seconds.
python -m framefinder index ~/Movies/my_videos
python -m framefinder index ~/Movies/my_videos --every 1     # denser, slower

# Search
python -m framefinder search "a dog running on a beach"
python -m framefinder search "two people shaking hands" -n 20 --open
```

Output looks like:

```
 1. 0.312  01:14  /Users/you/Movies/my_videos/trip.mp4
 2. 0.301  00:46  /Users/you/Movies/my_videos/birthday.mov
```

Open the video at that timestamp, or browse `results/<your-query>/results.html`.

### Options

| Flag | Command | Meaning |
|---|---|---|
| `--every S` | index | Seconds between sampled frames (default 2) |
| `--force` | index | Rebuild the index from scratch |
| `-n N` | search | Number of results (default 10) |
| `--gap S` | search | Minimum seconds between hits in the same video (default 6) |
| `--open` | search | Open the contact sheet in your browser |
| `--index-dir`, `--results-dir` | both | Where to keep the index / results |

## Tips

- CLIP scores are small numbers (about 0.2 to 0.35 is typical); look at ranking, not absolute values.
- Short, concrete descriptions work best: "a red car at night", "someone cooking in a kitchen".
  It matches what is visible in a frame, not audio, motion, or text on screen.
- A 1-hour video at the default rate is about 1,800 frames. Expect a few minutes to index on an M2 with the base model.

## Layout

```
framefinder/
  setup.sh              one-time setup
  requirements.txt
  framefinder/
    embedder.py         MLX CLIP image/text embeddings
    frames.py           video scanning and frame sampling (OpenCV)
    store.py            on-disk index
    search.py           ranking, thumbnails, contact sheet
    cli.py              `index` and `search` commands
  tests/test_logic.py   tests that run without MLX
```
