"""On-disk index: one manifest.json plus one .npz of embeddings per video."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

DEFAULT_INDEX_DIR = Path(".framefinder_index")


class Index:
    def __init__(self, path: str | Path = DEFAULT_INDEX_DIR):
        self.dir = Path(path)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = {"model": None, "videos": {}}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())

    # ---- bookkeeping -------------------------------------------------
    @staticmethod
    def _signature(video: Path) -> dict:
        st = video.stat()
        return {"size": st.st_size, "mtime": int(st.st_mtime)}

    def check_model(self, model_name: str) -> None:
        current = self.manifest.get("model")
        if current not in (None, model_name):
            raise SystemExit(
                f"This index was built with model '{current}', but you are using '{model_name}'. "
                "Re-run `index --force` to rebuild it."
            )
        self.manifest["model"] = model_name

    def is_current(self, video: Path, every_s: float) -> bool:
        entry = self.manifest["videos"].get(str(video.resolve()))
        if not entry:
            return False
        sig = self._signature(video)
        return (
            entry["size"] == sig["size"]
            and entry["mtime"] == sig["mtime"]
            and entry["every_s"] == every_s
            and (self.dir / entry["file"]).exists()
        )

    def add(self, video: Path, times: List[float], embs: np.ndarray, every_s: float) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        key = str(video.resolve())
        fname = hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz"
        np.savez(self.dir / fname, times=np.asarray(times, dtype=np.float32), embs=embs.astype(np.float16))
        self.manifest["videos"][key] = {
            **self._signature(video),
            "every_s": every_s,
            "file": fname,
            "frames": len(times),
        }
        self.save()

    def prune_missing(self) -> int:
        gone = [k for k in self.manifest["videos"] if not Path(k).exists()]
        for k in gone:
            f = self.dir / self.manifest["videos"][k]["file"]
            f.unlink(missing_ok=True)
            del self.manifest["videos"][k]
        if gone:
            self.save()
        return len(gone)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    # ---- reading -----------------------------------------------------
    def load_all(self) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        all_embs, meta = [], []
        for video, entry in self.manifest["videos"].items():
            if not Path(video).exists():
                continue
            data = np.load(self.dir / entry["file"])
            all_embs.append(data["embs"].astype(np.float32))
            meta.extend((video, float(t)) for t in data["times"])
        if not all_embs:
            return np.zeros((0, 1), dtype=np.float32), []
        return np.concatenate(all_embs, axis=0), meta
