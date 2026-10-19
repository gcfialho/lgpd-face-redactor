"""Reading and writing frames for videos and still images, plus optional audio muxing."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    is_image: bool
    frame_count: int
    fps: float
    width: int
    height: int


def probe(path: Path) -> MediaInfo:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return MediaInfo(path, True, 1, 0.0, image.shape[1], image.shape[0])

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = MediaInfo(
        path,
        False,
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        cap.get(cv2.CAP_PROP_FPS) or 30.0,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def frames(info: MediaInfo) -> Iterator[np.ndarray]:
    if info.is_image:
        yield cv2.imread(str(info.path))
        return
    cap = cv2.VideoCapture(str(info.path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class Writer:
    def __init__(self, info: MediaInfo, output: Path):
        self.info, self.output = info, output
        self._video: cv2.VideoWriter | None = None
        if not info.is_image:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video = cv2.VideoWriter(str(output), fourcc, info.fps, (info.width, info.height))
            if not self._video.isOpened():
                raise ValueError(f"Cannot write video: {output}")

    def write(self, frame: np.ndarray) -> None:
        if self._video is None:
            cv2.imwrite(str(self.output), frame)
        else:
            self._video.write(frame)

    def close(self) -> None:
        if self._video is not None:
            self._video.release()


def mux_audio(source: Path, redacted: Path) -> bool:
    """Copy the original audio track into the redacted video. Returns False if ffmpeg is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    tmp = redacted.with_name(redacted.stem + ".audio" + redacted.suffix)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(redacted), "-i", str(source),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", str(tmp)]  # fmt: skip
    subprocess.run(cmd, check=True)
    tmp.replace(redacted)
    return True
