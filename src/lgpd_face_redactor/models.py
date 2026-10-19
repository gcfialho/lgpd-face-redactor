"""Model weights: downloaded once from the OpenCV Zoo, verified by SHA-256, cached locally."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    license: str


YUNET = ModelSpec(
    name="YuNet face detector (2023mar)",
    filename="face_detection_yunet_2023mar.onnx",
    url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    license="MIT",
)

SFACE = ModelSpec(
    name="SFace face recognizer (2021dec)",
    filename="face_recognition_sface_2021dec.onnx",
    url="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    license="Apache-2.0",
)


def cache_dir() -> Path:
    root = (
        os.getenv("LGPD_REDACTOR_MODELS")
        or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "lgpd-face-redactor"
    )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(spec: ModelSpec) -> Path:
    """Return a verified local copy of the model, downloading it if needed."""
    path = cache_dir() / spec.filename
    if path.exists() and sha256_of(path) == spec.sha256:
        return path

    log.info("Downloading %s ...", spec.name)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(spec.url, tmp)  # fixed https URL; the checksum is verified below
    actual = sha256_of(tmp)
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, got {actual}")
    tmp.replace(path)
    return path
