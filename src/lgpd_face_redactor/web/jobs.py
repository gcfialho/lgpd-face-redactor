"""Job lifecycle for the web app: upload, analyse, pick people, render, download, delete.

Everything a job touches lives in one private temporary folder (mode 0700): the upload,
the face thumbnails shown in the gallery and the redacted copy. The folder is removed
when the user deletes the job or when it expires, whichever comes first.

Jobs run one at a time on a single worker thread. OpenCV's DNN networks are not
thread-safe, and serialising is the right trade-off for a local, single-user tool.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from lgpd_face_redactor import audit, media
from lgpd_face_redactor.matching import ReferenceGallery
from lgpd_face_redactor.people import Person, gallery_embeddings, group_people
from lgpd_face_redactor.pipeline import Analysis, RedactionConfig, analyze, assess, config_for, render_file
from lgpd_face_redactor.types import FaceDetector, FaceEmbedder, Style

ALLOWED_SUFFIXES = media.IMAGE_SUFFIXES | {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CHUNK = 1 << 20


class Status(StrEnum):
    QUEUED = "queued"
    ANALYSING = "analysing"
    READY = "ready"  # people found, waiting for the reviewer's selection
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


class UploadError(ValueError):
    pass


@dataclass
class WebJob:
    id: str
    folder: Path
    source: Path
    original_name: str
    created: float = field(default_factory=time.time)
    status: Status = Status.QUEUED
    progress: float = 0.0
    message: str = "Waiting to start"
    error: str | None = None
    info: media.MediaInfo | None = None
    config: RedactionConfig = field(default_factory=RedactionConfig)
    analysis: Analysis | None = None
    people: list[Person] = field(default_factory=list)
    thumbnails: dict[int, Path] = field(default_factory=dict)
    preselected: set[int] = field(default_factory=set)
    selected: set[int] = field(default_factory=set)
    output: Path | None = None
    preview: Path | None = None  # browser-playable copy of the output, when ffmpeg is available
    record: dict | None = None

    @property
    def busy(self) -> bool:
        return self.status in {Status.QUEUED, Status.ANALYSING, Status.RENDERING}


class JobStore:
    def __init__(
        self,
        detector_factory: Callable[[], FaceDetector],
        embedder_factory: Callable[[], FaceEmbedder],
        root: Path | None = None,
        ttl_seconds: int = 3600,
        max_upload_bytes: int = 500 * CHUNK,
        inline: bool = False,  # run jobs synchronously (tests)
    ):
        self._detector_factory, self._embedder_factory = detector_factory, embedder_factory
        self._detector: FaceDetector | None = None
        self._embedder: FaceEmbedder | None = None
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.max_upload_bytes = max_upload_bytes
        self._jobs: dict[str, WebJob] = {}
        self._lock = threading.Lock()
        self._inline = inline
        self._executor = None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix="redactor")

    # ---- lifecycle -------------------------------------------------------------------------

    def create(self, upload: BinaryIO, filename: str, references: list[tuple[BinaryIO, str]], style: Style) -> WebJob:
        self.expire()
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UploadError(f"Unsupported file type {suffix or '(none)'}")

        folder = Path(tempfile.mkdtemp(prefix="lgpd-redactor-", dir=self.root))
        os.chmod(folder, 0o700)
        job_id = uuid.uuid4().hex
        try:
            source = self._save(upload, folder / f"input{suffix}")
            reference_paths = [
                self._save(f, folder / f"reference-{i}{Path(name).suffix.lower()}")
                for i, (f, name) in enumerate(references)
            ]
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

        job = WebJob(job_id, folder, source, Path(filename).name)
        job.config = replace(job.config, style=style)
        with self._lock:
            self._jobs[job_id] = job
        self._submit(self._analyse, job, reference_paths)
        return job

    def get(self, job_id: str) -> WebJob | None:
        self.expire()
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.folder, ignore_errors=True)
        return True

    def expire(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            stale = [j.id for j in self._jobs.values() if j.created < cutoff and not j.busy]
        for job_id in stale:
            self.delete(job_id)

    def shutdown(self) -> None:
        for job_id in list(self._jobs):
            self.delete(job_id)
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def render(self, job: WebJob, selected: set[int], style: Style) -> None:
        if job.status not in {Status.READY, Status.DONE}:
            raise ValueError("The job is not ready to render")
        job.selected = {pid for pid in selected if any(p.id == pid and p.comparable for p in job.people)}
        job.config = replace(job.config, style=style)
        job.status, job.progress, job.message = Status.RENDERING, 0.0, "Queued for rendering"
        self._submit(self._render, job)

    # ---- work ------------------------------------------------------------------------------

    def _submit(self, fn: Callable, *args) -> Future | None:
        if self._inline:
            fn(*args)
            return None
        return self._executor.submit(fn, *args)

    def _models(self) -> tuple[FaceDetector, FaceEmbedder]:
        if self._detector is None:
            self._detector = self._detector_factory()
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._detector, self._embedder

    def _analyse(self, job: WebJob, references: list[Path]) -> None:
        try:
            job.status, job.message = Status.ANALYSING, "Loading models"
            detector, embedder = self._models()
            job.info = media.probe(job.source)
            job.config = config_for(job.info, job.config)

            reference = ReferenceGallery()
            if references:
                from lgpd_face_redactor.vision import reference_embedding

                for path in references:
                    reference.add(reference_embedding(path, detector, embedder))

            total = max(1, job.info.frame_count)

            def progress(index: int) -> None:
                job.progress = min(0.99, (index + 1) / total)
                job.message = f"Finding faces: frame {index + 1} of {total}"

            job.analysis = analyze(media.frames(job.info), detector, embedder, job.config, progress)
            job.people = group_people(job.analysis.tracks)
            job.message = "Preparing the gallery"
            job.thumbnails = self._thumbnails(job)
            if len(reference):
                threshold = job.config.policy.threshold
                job.preselected = {
                    p.id
                    for p in job.people
                    if p.comparable and max(reference.best_similarity(e) for e in p.embeddings) >= threshold
                }
            for path in references:  # reference photos are no longer needed
                path.unlink(missing_ok=True)
            job.status, job.progress = Status.READY, 1.0
            job.message = f"Found {len(job.people)} people in {job.analysis.frame_count} frames"
        except Exception as exc:  # surfaced to the user; the job folder is still deleted on expiry
            job.status, job.error, job.message = Status.FAILED, str(exc), "Analysis failed"

    def _thumbnails(self, job: WebJob) -> dict[int, Path]:
        import cv2

        wanted: dict[int, list[Person]] = {}
        for person in job.people:
            wanted.setdefault(person.best.frame, []).append(person)
        out: dict[int, Path] = {}
        thumbs = job.folder / "thumbs"
        thumbs.mkdir()
        for index, frame in enumerate(media.frames(job.info)):
            for person in wanted.get(index, []):
                box = person.best.box.expanded(1.3)
                x1, y1 = max(0, int(box.x)), max(0, int(box.y))
                crop = frame[y1 : int(box.y + box.h), x1 : int(box.x + box.w)]
                if crop.size:
                    path = thumbs / f"{person.id}.jpg"
                    cv2.imwrite(str(path), crop)
                    out[person.id] = path
            if len(out) == len(job.people):
                break
        return out

    def _render(self, job: WebJob) -> None:
        try:
            assert job.analysis is not None and job.info is not None
            gallery = ReferenceGallery(gallery_embeddings(job.people, job.selected))
            assess(job.analysis, gallery, job.config)
            job.message, job.progress = "Rendering the redacted copy", 0.5
            suffix = job.source.suffix if job.info.is_image else ".mp4"
            job.output = job.folder / f"redacted{suffix}"
            render_file(job.source, job.info, job.analysis, job.output, job.config)
            job.preview = job.output if job.info.is_image else media.to_browser_mp4(job.output)
            mode = "selected-people" if job.selected else "blur-all"
            job.record = audit.build(
                job.source, job.output, job.info, job.analysis, job.config, self._detector, False,
                mode=mode, kept_people=len(job.selected), original_name=job.original_name,
            )  # fmt: skip
            job.status, job.progress, job.message = Status.DONE, 1.0, "Done"
        except Exception as exc:
            job.status, job.error, job.message = Status.FAILED, str(exc), "Rendering failed"

    def _save(self, upload: BinaryIO, path: Path) -> Path:
        written = 0
        with path.open("wb") as out:
            while chunk := upload.read(CHUNK):
                written += len(chunk)
                if written > self.max_upload_bytes:
                    raise UploadError(f"File larger than {self.max_upload_bytes // CHUNK} MB")
                out.write(chunk)
        if written == 0:
            raise UploadError("Empty upload")
        return path
