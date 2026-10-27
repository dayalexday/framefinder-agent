"""CLIP embeddings on Apple Silicon via MLX.

Uses Apple's reference MLX CLIP code (downloaded into vendor/mlx_clip by setup.sh)
and the converted weights in ./mlx_model. MLX is imported lazily so the rest of the
project (indexing logic, tests) can be imported on machines without it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "mlx_clip"
DEFAULT_MODEL_DIR = Path(os.environ.get("FRAMEFINDER_MODEL", ROOT / "mlx_model"))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class ClipEmbedder:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, batch_size: int = 32):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists() or not VENDOR.exists():
            raise SystemExit(
                f"Model not found at {model_dir} (or vendor code missing). Run ./setup.sh first."
            )
        # The vendored example uses flat imports (`from model import ...`).
        sys.path.insert(0, str(VENDOR))
        import mlx.core as mx  # noqa: WPS433 (lazy on purpose)
        import clip as mlx_clip_example

        self._mx = mx
        self.model, self.tokenizer, self.img_processor = mlx_clip_example.load(str(model_dir))
        self.batch_size = batch_size
        self.name = model_dir.name

    def embed_text(self, text: str) -> np.ndarray:
        out = self.model(input_ids=self.tokenizer([text]))
        emb = out.text_embeds
        self._mx.eval(emb)
        return _l2_normalize(np.array(emb, dtype=np.float32))[0]

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        chunks = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            out = self.model(pixel_values=self.img_processor(batch))
            emb = out.image_embeds
            self._mx.eval(emb)
            chunks.append(np.array(emb, dtype=np.float32))
        return _l2_normalize(np.concatenate(chunks, axis=0))
