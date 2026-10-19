"""Links detections across frames so decisions are made per person, not per frame.

Greedy association on IoU, gated by appearance when embeddings are available: two
boxes that overlap but look like different people start separate tracks. That gate
matters here, because an identity switch onto the data subject's track is exactly
how a third party would leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lgpd_face_redactor.matching import normalize
from lgpd_face_redactor.types import Detection, Observation, Track


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_gap: int = 10  # frames a track may go undetected before it is closed
    appearance_threshold: float = 0.2  # min cosine similarity between consecutive embeddings of one track


@dataclass
class _ActiveTrack:
    track: Track
    last_embedding: np.ndarray | None = None


@dataclass
class IouTracker:
    config: TrackerConfig = field(default_factory=TrackerConfig)
    _active: list[_ActiveTrack] = field(default_factory=list)
    _finished: list[Track] = field(default_factory=list)
    _next_id: int = 0

    def update(
        self,
        frame: int,
        detections: list[Detection],
        embeddings: list[np.ndarray | None],
    ) -> None:
        self._close_stale(frame)

        candidates = []
        for t_idx, active in enumerate(self._active):
            for d_idx, det in enumerate(detections):
                iou = active.track.last_box.iou(det.box)
                if iou < self.config.iou_threshold:
                    continue
                if not self._same_appearance(active.last_embedding, embeddings[d_idx]):
                    continue
                candidates.append((iou, t_idx, d_idx))

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, t_idx, d_idx in sorted(candidates, reverse=True):
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            self._append(self._active[t_idx], frame, detections[d_idx], embeddings[d_idx])

        for d_idx, det in enumerate(detections):
            if d_idx not in used_dets:
                active = _ActiveTrack(Track(id=self._next_id))
                self._next_id += 1
                self._append(active, frame, det, embeddings[d_idx])
                self._active.append(active)

    def finish(self) -> list[Track]:
        self._finished.extend(a.track for a in self._active)
        self._active = []
        return sorted(self._finished, key=lambda t: t.id)

    def _close_stale(self, frame: int) -> None:
        still_active = []
        for active in self._active:
            if frame - active.track.last_frame > self.config.max_gap:
                self._finished.append(active.track)
            else:
                still_active.append(active)
        self._active = still_active

    def _same_appearance(self, previous: np.ndarray | None, current: np.ndarray | None) -> bool:
        if previous is None or current is None:
            return True
        return float(previous @ normalize(current)) >= self.config.appearance_threshold

    @staticmethod
    def _append(active: _ActiveTrack, frame: int, det: Detection, embedding: np.ndarray | None) -> None:
        unit = normalize(embedding) if embedding is not None else None
        stored = unit.astype(np.float16) if unit is not None else None
        active.track.observations.append(Observation(frame, det.box, det.score, embedding=stored))
        if unit is not None:
            active.last_embedding = unit
