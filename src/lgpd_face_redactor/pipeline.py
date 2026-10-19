"""Two-pass redaction.

Pass 1 detects, embeds and tracks every face. Decisions are then made per track, using
the whole track, which a single streaming pass cannot do. Pass 2 renders the plan.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lgpd_face_redactor.matching import ReferenceGallery
from lgpd_face_redactor.policy import Decision, PolicyConfig, RenderPlan, TrackDecision, decide, plan_render
from lgpd_face_redactor.tracking import IouTracker, TrackerConfig
from lgpd_face_redactor.types import FaceDetector, FaceEmbedder, Style, Track

log = logging.getLogger(__name__)


@dataclass
class RedactionConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    style: Style = Style.PIXELATE
    blocks: int = 8
    keep_audio: bool = False


@dataclass
class Analysis:
    tracks: list[Track]
    decisions: dict[int, TrackDecision]
    plan: RenderPlan
    frame_count: int
    faces_seen: int


def analyze(
    frames: Iterable[np.ndarray],
    detector: FaceDetector,
    embedder: FaceEmbedder | None,
    gallery: ReferenceGallery,
    config: RedactionConfig,
) -> Analysis:
    tracker = IouTracker(config.tracker)
    frame_count = faces_seen = 0
    for index, frame in enumerate(frames):
        detections = detector.detect(frame)
        faces_seen += len(detections)
        embeddings = [embedder.embed(frame, d) if embedder else None for d in detections]
        similarities = [gallery.best_similarity(e) if e is not None and len(gallery) else None for e in embeddings]
        tracker.update(index, detections, similarities, embeddings)
        frame_count = index + 1

    tracks = tracker.finish()
    has_reference = len(gallery) > 0
    decisions = {t.id: decide(t, config.policy, has_reference) for t in tracks}
    plan = plan_render(tracks, decisions, config.policy, frame_count)
    return Analysis(tracks, decisions, plan, frame_count, faces_seen)


def render(frames: Iterable[np.ndarray], plan: RenderPlan, config: RedactionConfig) -> Iterable[np.ndarray]:
    from lgpd_face_redactor.render import obscure

    for index, frame in enumerate(frames):
        boxes = plan.blur_boxes.get(index, [])
        yield obscure(frame, boxes, config.style, config.blocks) if boxes else frame


@dataclass
class Job:
    """Everything needed to analyse and render one input; built once, rendered as often as the reviewer likes."""

    source: Path
    info: object  # media.MediaInfo
    config: RedactionConfig
    detector: FaceDetector
    embedder: FaceEmbedder | None
    gallery: ReferenceGallery
    analysis: Analysis | None = None
    overrides: dict[int, Decision] = field(default_factory=dict)


def prepare(
    source: Path,
    references: list[Path],
    config: RedactionConfig,
    detector: FaceDetector | None = None,
    embedder: FaceEmbedder | None = None,
) -> Job:
    from lgpd_face_redactor import media
    from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector, reference_embedding

    info = media.probe(source)
    if info.is_image:
        # A still image has no history to confirm a match, so one frame is enough.
        config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, min_frames=1, pad_frames=0))

    detector = detector or YuNetDetector()
    if references and embedder is None:
        embedder = SFaceEmbedder()
    gallery = ReferenceGallery([reference_embedding(r, detector, embedder) for r in references])
    return Job(source, info, config, detector, embedder, gallery)


def run_analysis(job: Job) -> Analysis:
    from lgpd_face_redactor import media

    job.analysis = analyze(media.frames(job.info), job.detector, job.embedder, job.gallery, job.config)
    return job.analysis


def apply_overrides(job: Job, overrides: dict[int, Decision]) -> Analysis:
    """A reviewer's decisions replace the automatic ones and are recorded as such in the audit."""
    assert job.analysis is not None, "run_analysis first"
    analysis = job.analysis
    decisions = dict(analysis.decisions)
    for track_id, decision in overrides.items():
        auto = decisions[track_id]
        if decision is not auto.decision:
            decisions[track_id] = dataclasses.replace(
                auto,
                decision=decision,
                reason=f"manual override by reviewer (automatic: {auto.decision}, {auto.reason})",
            )
    job.overrides = dict(overrides)
    plan = plan_render(analysis.tracks, decisions, job.config.policy, analysis.frame_count)
    job.analysis = dataclasses.replace(analysis, decisions=decisions, plan=plan)
    return job.analysis


def render_file(job: Job, output: Path) -> dict:
    from lgpd_face_redactor import audit, media

    assert job.analysis is not None, "run_analysis first"
    writer = media.Writer(job.info, output)
    try:
        for frame in render(media.frames(job.info), job.analysis.plan, job.config):
            writer.write(frame)
    finally:
        writer.close()

    audio = False
    if job.config.keep_audio and not job.info.is_image:
        audio = media.mux_audio(job.source, output)
        if not audio:
            log.warning("ffmpeg not found: the output has no audio")

    analysis = job.analysis
    kept = sum(d.decision is Decision.KEEP for d in analysis.decisions.values())
    log.info(
        "%d faces in %d tracks; %d kept visible, the rest blurred", analysis.faces_seen, len(analysis.tracks), kept
    )
    return audit.build(job, output, audio)


def redact_file(
    source: Path,
    output: Path,
    references: list[Path],
    config: RedactionConfig,
    detector: FaceDetector | None = None,
    embedder: FaceEmbedder | None = None,
) -> tuple[Analysis, dict]:
    job = prepare(source, references, config, detector, embedder)
    run_analysis(job)
    record = render_file(job, output)
    return job.analysis, record
