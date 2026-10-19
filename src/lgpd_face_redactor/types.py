"""Plain data types shared by every stage. No OpenCV here, so the core logic is testable anywhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np


class Style(StrEnum):
    PIXELATE = "pixelate"
    BLUR = "blur"
    BOX = "box"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: Box) -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def expanded(self, factor: float) -> Box:
        """Grow the box around its centre. Detector boxes are tight; hair, ears and chin identify people too."""
        dw, dh = self.w * (factor - 1) / 2, self.h * (factor - 1) / 2
        return Box(self.x - dw, self.y - dh, self.w * factor, self.h * factor)

    @staticmethod
    def lerp(a: Box, b: Box, t: float) -> Box:
        return Box(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.w + (b.w - a.w) * t,
            a.h + (b.h - a.h) * t,
        )


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    # Raw detector row (box, 5 landmarks, score). Needed to align the face before embedding.
    raw: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass
class Observation:
    """One detection assigned to a track.

    `embedding` is the L2-normalised face embedding (float16 to keep long videos in memory),
    or None when the face could not be aligned. `similarity` is filled in later, against
    whichever people the user decided to keep.
    """

    frame: int
    box: Box
    score: float
    similarity: float | None = None
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class Track:
    id: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame

    @property
    def last_box(self) -> Box:
        return self.observations[-1].box

    def similarities(self) -> list[float]:
        return [o.similarity for o in self.observations if o.similarity is not None]

    def embeddings(self) -> list[np.ndarray]:
        return [o.embedding.astype(np.float32) for o in self.observations if o.embedding is not None]

    def mean_embedding(self) -> np.ndarray | None:
        vectors = self.embeddings()
        if not vectors:
            return None
        mean = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm else None

    def frames(self) -> set[int]:
        return {o.frame for o in self.observations}


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class FaceEmbedder(Protocol):
    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray | None: ...
