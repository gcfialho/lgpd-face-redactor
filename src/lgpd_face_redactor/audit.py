"""Audit record for each run, so a controller can show what was done and why (LGPD art. 37, accountability).

It never contains biometric templates, face images or the reference photos: only
hashes, settings, and per-track decisions with their similarity statistics.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lgpd_face_redactor import __version__, models
from lgpd_face_redactor.policy import Decision


def build(
    source: Path,
    output: Path,
    info,
    analysis,
    config,
    detector,
    audio: bool,
    mode: str,
    kept_people: int,
    original_name: str | None = None,
) -> dict[str, Any]:
    """`mode` is "keep-references" (CLI photos), "selected-people" (web gallery) or "blur-all"."""
    decisions = [analysis.decisions[t.id] for t in analysis.tracks]
    return {
        "tool": {"name": "lgpd-face-redactor", "version": __version__},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {
            "file": original_name or source.name,
            "sha256": models.sha256_of(source),
            "frames": analysis.frame_count,
            "fps": info.fps,
            "resolution": [info.width, info.height],
        },
        "output": {"file": output.name, "sha256": models.sha256_of(output), "audio_kept": audio},
        "models": [{"name": m.name, "sha256": m.sha256, "license": m.license} for m in (models.YUNET, models.SFACE)],
        "settings": {
            "mode": mode,
            "people_kept_visible": kept_people if mode != "blur-all" else 0,
            "detector_score_threshold": getattr(detector, "score_threshold", None),
            "policy": dataclasses.asdict(config.policy),
            "tracker": dataclasses.asdict(config.tracker),
            "style": str(config.style),
        },
        "summary": {
            "faces_detected": analysis.faces_seen,
            "tracks": len(decisions),
            "tracks_kept": sum(d.decision is Decision.KEEP for d in decisions),
            "tracks_blurred": sum(d.decision is Decision.BLUR for d in decisions),
            "frames_with_redaction": len(analysis.plan.blur_boxes),
        },
        "tracks": [
            {
                **dataclasses.asdict(analysis.decisions[t.id]),
                "decision": str(analysis.decisions[t.id].decision),
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
            }
            for t in analysis.tracks
        ],
    }
