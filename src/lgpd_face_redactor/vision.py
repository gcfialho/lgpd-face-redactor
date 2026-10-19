"""OpenCV wrappers for detection (YuNet) and embedding (SFace)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lgpd_face_redactor import models
from lgpd_face_redactor.types import Box, Detection, FaceDetector, FaceEmbedder


class YuNetDetector:
    # A low score threshold on purpose: a missed face is a leak, a spurious box only blurs background.
    def __init__(self, model_path: Path | None = None, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        path = model_path or models.ensure(models.YUNET)
        self.score_threshold = score_threshold
        self._net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, nms_threshold, 5000)

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        self._net.setInputSize((width, height))
        _, faces = self._net.detect(image)
        if faces is None:
            return []
        return [Detection(Box(*map(float, row[:4])), float(row[14]), raw=row) for row in faces]


class SFaceEmbedder:
    def __init__(self, model_path: Path | None = None):
        path = model_path or models.ensure(models.SFACE)
        self._net = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, image: np.ndarray, detection: Detection) -> np.ndarray | None:
        if detection.raw is None:
            return None
        try:
            aligned = self._net.alignCrop(image, detection.raw)
            return self._net.feature(aligned).reshape(-1).astype(np.float32)
        except cv2.error:
            # Faces cut by the frame edge can fail to align. The policy treats them as unknown, i.e. blurred.
            return None


def reference_embedding(path: Path, detector: FaceDetector, embedder: FaceEmbedder) -> np.ndarray:
    """Embed a reference photo. It must contain exactly one face, otherwise we cannot know who to keep."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read reference image: {path}")
    faces = detector.detect(image)
    if len(faces) != 1:
        raise ValueError(f"Reference {path} must contain exactly one face, found {len(faces)}")
    embedding = embedder.embed(image, faces[0])
    if embedding is None:
        raise ValueError(f"Could not align the face in {path}")
    return embedding
