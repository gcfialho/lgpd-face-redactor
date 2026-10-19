"""Fail-safe redaction policy.

The two possible errors are not symmetric. Blurring the data subject by mistake makes
the copy less useful; leaving a third party visible is a data leak. So a face is shown
only when the evidence is strong at both levels:

1. Track level: the person must match the reference consistently over time, not in a
   lucky frame (median similarity, share of matching frames, minimum number of frames).
2. Frame level: even inside a kept track, any frame whose own similarity is low (or
   could not be measured) is blurred. That contains identity switches in the tracker.

Everything else is blurred, including the frames around a blurred track where the
detector may have missed the face.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import StrEnum

from lgpd_face_redactor.types import Box, Track


class Decision(StrEnum):
    KEEP = "keep"
    BLUR = "blur"


@dataclass(frozen=True)
class PolicyConfig:
    # Cosine similarity for SFace. OpenCV reports 0.363 as its LFW operating point; we are
    # stricter by default because a false match is the costly error. Calibrate with `eval/`.
    threshold: float = 0.42
    frame_threshold: float = 0.36  # per-frame floor inside a kept track
    min_matched_share: float = 0.6  # share of the track's frames that must clear `threshold`
    min_frames: int = 5  # a still image is evaluated with min_frames=1
    pad_frames: int = 3  # extra frames blurred before/after each blurred appearance
    box_expansion: float = 1.35


@dataclass(frozen=True)
class TrackDecision:
    track_id: int
    decision: Decision
    reason: str
    frames: int
    median_similarity: float | None
    matched_share: float | None


@dataclass
class RenderPlan:
    """Boxes to blur, per frame index."""

    blur_boxes: dict[int, list[Box]] = field(default_factory=dict)

    def add(self, frame: int, box: Box) -> None:
        self.blur_boxes.setdefault(frame, []).append(box)


def decide(track: Track, config: PolicyConfig, has_reference: bool) -> TrackDecision:
    frames = len(track.observations)
    if not has_reference:
        return TrackDecision(track.id, Decision.BLUR, "no reference: blur everyone", frames, None, None)

    sims = track.similarities()
    if not sims:
        return TrackDecision(track.id, Decision.BLUR, "face could not be compared", frames, None, None)

    median = statistics.median(sims)
    share = sum(s >= config.threshold for s in sims) / frames  # frames without a similarity count against

    if frames < config.min_frames:
        reason, decision = f"too short to confirm ({frames} < {config.min_frames} frames)", Decision.BLUR
    elif median < config.threshold:
        reason, decision = f"median similarity {median:.3f} below {config.threshold}", Decision.BLUR
    elif share < config.min_matched_share:
        reason, decision = f"only {share:.0%} of frames match (needs {config.min_matched_share:.0%})", Decision.BLUR
    else:
        reason, decision = "consistent match with the reference", Decision.KEEP

    return TrackDecision(track.id, decision, reason, frames, round(median, 4), round(share, 4))


def plan_render(
    tracks: list[Track], decisions: dict[int, TrackDecision], config: PolicyConfig, frame_count: int
) -> RenderPlan:
    plan = RenderPlan()
    for track in tracks:
        if decisions[track.id].decision is Decision.KEEP:
            for obs in track.observations:
                if obs.similarity is None or obs.similarity < config.frame_threshold:
                    plan.add(obs.frame, obs.box.expanded(config.box_expansion))
            continue

        obs = track.observations
        for i, current in enumerate(obs):
            plan.add(current.frame, current.box.expanded(config.box_expansion))
            # Fill detector misses between two sightings of the same blurred person.
            if i + 1 < len(obs):
                nxt = obs[i + 1]
                gap = nxt.frame - current.frame
                for f in range(current.frame + 1, nxt.frame):
                    t = (f - current.frame) / gap
                    plan.add(f, Box.lerp(current.box, nxt.box, t).expanded(config.box_expansion))
        # Hold the first and last box a few frames longer.
        for f in range(max(0, obs[0].frame - config.pad_frames), obs[0].frame):
            plan.add(f, obs[0].box.expanded(config.box_expansion))
        for f in range(obs[-1].frame + 1, min(frame_count, obs[-1].frame + config.pad_frames + 1)):
            plan.add(f, obs[-1].box.expanded(config.box_expansion))
    return plan
