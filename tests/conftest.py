"""Scripted fakes: deterministic detections and identity-coded embeddings, so tests need no faces or models."""

from __future__ import annotations

import numpy as np
import pytest

from lgpd_face_redactor.types import Box, Detection

DIM = 128


def identity_vector(seed: int, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A fixed random direction per identity, optionally jittered to mimic frame-to-frame variation."""
    base = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    if noise and rng is not None:
        base = base + noise * rng.standard_normal(DIM).astype(np.float32) * np.linalg.norm(base) / np.sqrt(DIM)
    return base / np.linalg.norm(base)


class ScriptedDetector:
    """frames[i] is a list of (box, identity) pairs for frame i."""

    def __init__(self, frames: list[list[tuple[Box, int]]]):
        self.frames = frames
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        faces = self.frames[self.calls]
        self.calls += 1
        return [Detection(box, 0.9, raw=np.array([identity], dtype=np.float32)) for box, identity in faces]


class IdentityEmbedder:
    """Embeds a detection as its identity's vector plus noise. identity < 0 means 'cannot align'."""

    def __init__(self, noise: float = 0.15, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray | None:
        identity = int(detection.raw[0])
        if identity < 0:
            return None
        return identity_vector(identity, self.noise, self.rng)


def walk(start: Box, dx: float, n: int) -> list[Box]:
    return [Box(start.x + dx * i, start.y, start.w, start.h) for i in range(n)]


@pytest.fixture
def blank_frames():
    def make(n: int) -> list[np.ndarray]:
        return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]

    return make
