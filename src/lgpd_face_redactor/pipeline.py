"""Redaction in three steps, so the people to keep can be chosen after seeing who is there.

1. `analyze`: detect, embed and track every face (one pass over the frames).
2. `assess`: compare every sighting with the people to keep, decide per track, plan the blur.
   It is cheap, so it can be re-run for each selection without touching the video again.
3. `render`: apply the plan (second pass over the frames).

The CLI keeps people given as reference photos; the web app keeps the people a reviewer
ticks in the gallery. Both feed the same `ReferenceGallery` and the same fail-safe policy.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lgpd_face_redactor.matching import ReferenceGallery
from lgpd_face_redactor.policy import Decision, PolicyConfig, RenderPlan, TrackDecision, decide, plan_render
from lgpd_face_redactor.tracking import IouTracker, TrackerConfig
from lgpd_face_redactor.types import FaceDetector, FaceEmbedder, Style, Track

log = logging.getLogger(__name__)

Progress = Callable[[int], None]  # called with the index of each processed frame


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
    frame_count: int
    faces_seen: int
    decisions: dict[int, TrackDecision] = field(default_factory=dict)
    plan: RenderPlan = field(default_factory=RenderPlan)


def analyze(
    frames: Iterable[np.ndarray],
    detector: FaceDetector,
    embedder: FaceEmbedder | None,
    config: RedactionConfig,
    progress: Progress | None = None,
) -> Analysis:
    tracker = IouTracker(config.tracker)
    frame_count = faces_seen = 0
    for index, frame in enumerate(frames):
        detections = detector.detect(frame)
        faces_seen += len(detections)
        embeddings = [embedder.embed(frame, d) if embedder else None for d in detections]
        tracker.update(index, detections, embeddings)
        frame_count = index + 1
        if progress:
            progress(index)
    return Analysis(tracker.finish(), frame_count, faces_seen)


def assess(analysis: Analysis, gallery: ReferenceGallery, config: RedactionConfig) -> Analysis:
    """Score every sighting against the people to keep, then decide and plan. Mutates and returns `analysis`."""
    has_reference = len(gallery) > 0
    for track in analysis.tracks:
        for obs in track.observations:
            obs.similarity = (
                gallery.best_similarity(obs.embedding.astype(np.float32))
                if has_reference and obs.embedding is not None
                else None
            )
    analysis.decisions = {t.id: decide(t, config.policy, has_reference) for t in analysis.tracks}
    analysis.plan = plan_render(analysis.tracks, analysis.decisions, config.policy, analysis.frame_count)
    return analysis


def render(frames: Iterable[np.ndarray], plan: RenderPlan, config: RedactionConfig) -> Iterable[np.ndarray]:
    from lgpd_face_redactor.render import obscure

    for index, frame in enumerate(frames):
        boxes = plan.blur_boxes.get(index, [])
        yield obscure(frame, boxes, config.style, config.blocks) if boxes else frame


def config_for(info, config: RedactionConfig) -> RedactionConfig:
    """A still image has no history to confirm a match, so one frame is enough evidence."""
    if info.is_image:
        return dataclasses.replace(config, policy=dataclasses.replace(config.policy, min_frames=1, pad_frames=0))
    return config


def render_file(source: Path, info, analysis: Analysis, output: Path, config: RedactionConfig) -> bool:
    """Writes the redacted copy. Returns whether the audio track was kept."""
    from lgpd_face_redactor import media

    writer = media.Writer(info, output)
    try:
        for frame in render(media.frames(info), analysis.plan, config):
            writer.write(frame)
    finally:
        writer.close()

    if config.keep_audio and not info.is_image:
        if media.mux_audio(source, output):
            return True
        log.warning("ffmpeg not found: the output has no audio")
    return False


def redact_file(
    source: Path,
    output: Path,
    references: list[Path],
    config: RedactionConfig,
    detector: FaceDetector | None = None,
    embedder: FaceEmbedder | None = None,
) -> tuple[Analysis, dict]:
    """CLI path: keep the people in the reference photos, blur everyone else."""
    from lgpd_face_redactor import audit, media
    from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector, reference_embedding

    info = media.probe(source)
    config = config_for(info, config)
    detector = detector or YuNetDetector()
    if references and embedder is None:
        embedder = SFaceEmbedder()
    gallery = ReferenceGallery([reference_embedding(r, detector, embedder) for r in references])

    analysis = assess(analyze(media.frames(info), detector, embedder, config), gallery, config)
    audio = render_file(source, info, analysis, output, config)

    kept = sum(d.decision is Decision.KEEP for d in analysis.decisions.values())
    log.info(
        "%d faces in %d tracks; %d kept visible, the rest blurred", analysis.faces_seen, len(analysis.tracks), kept
    )
    mode = "keep-references" if len(gallery) else "blur-all"
    record = audit.build(
        source, output, info, analysis, config, detector, audio, mode=mode, kept_people=len(references)
    )
    return analysis, record
