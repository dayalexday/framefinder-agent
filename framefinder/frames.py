"""Frame sampling and extraction with OpenCV."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
from PIL import Image

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
MAX_SIDE = 512  # downscale before embedding; CLIP only needs 224px anyway


def find_videos(folder: str | Path) -> List[Path]:
    folder = Path(folder).expanduser()
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")
    )


def _to_pil(frame_bgr) -> Image.Image:
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def sample_frames(video: str | Path, every_s: float = 2.0) -> Iterator[Tuple[float, Image.Image]]:
    """Yield (timestamp_seconds, PIL image) roughly every `every_s` seconds."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = n_frames / fps if fps > 0 and n_frames > 0 else None

        t, misses = 0.0, 0
        while duration is None or t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if duration is not None or misses >= 3:
                    break
            else:
                misses = 0
                yield t, _to_pil(frame)
            t += every_s
    finally:
        cap.release()


def grab_frame(video: str | Path, t: float) -> Image.Image | None:
    """Fetch one frame at time t (full resolution) for thumbnails."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None
    finally:
        cap.release()
