"""Compares face embeddings with the data subject's reference embeddings."""

from __future__ import annotations

import numpy as np


def normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding")
    return vector / norm


class ReferenceGallery:
    """Embeddings of the people who must stay visible. Only vectors are kept, never images."""

    def __init__(self, embeddings: list[np.ndarray] | None = None):
        self._matrix = np.empty((0, 0), dtype=np.float32)
        for embedding in embeddings or []:
            self.add(embedding)

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def add(self, embedding: np.ndarray) -> None:
        vector = normalize(embedding)
        if len(self) == 0:
            self._matrix = vector[None, :]
        elif vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(f"Embedding size {vector.shape[0]} does not match gallery size {self._matrix.shape[1]}")
        else:
            self._matrix = np.vstack([self._matrix, vector])

    def best_similarity(self, embedding: np.ndarray) -> float:
        """Highest cosine similarity against any reference, in [-1, 1]."""
        if len(self) == 0:
            raise ValueError("The gallery is empty")
        return float(np.max(self._matrix @ normalize(embedding)))
