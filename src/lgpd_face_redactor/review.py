"""Human review UI: inspect every track, override decisions, render, download.

    uv run --extra review streamlit run src/lgpd_face_redactor/review.py

Runs locally. Uploaded files live in a temporary folder for the session only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from lgpd_face_redactor import media
from lgpd_face_redactor.pipeline import RedactionConfig, apply_overrides, prepare, render_file, run_analysis
from lgpd_face_redactor.policy import Decision, PolicyConfig
from lgpd_face_redactor.types import Style

st.set_page_config(page_title="LGPD Face Redactor", layout="wide")
st.title("LGPD Face Redactor — review")
st.caption("Every face is blurred except the people in the reference photos. Review each decision before export.")


def workdir() -> Path:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.TemporaryDirectory(prefix="lgpd-redactor-")
    return Path(st.session_state.workdir.name)


def save(upload, name: str) -> Path:
    path = workdir() / name
    path.write_bytes(upload.getbuffer())
    return path


def thumbnails(job) -> dict[int, object]:
    """One crop per track, from its most confident detection. Shown in this local session only."""
    wanted = {}
    for track in job.analysis.tracks:
        best = max(track.observations, key=lambda o: o.score)
        wanted.setdefault(best.frame, []).append((track.id, best.box.expanded(1.2)))
    crops = {}
    for index, frame in enumerate(media.frames(job.info)):
        for track_id, box in wanted.get(index, []):
            x1, y1 = max(0, int(box.x)), max(0, int(box.y))
            crop = frame[y1 : int(box.y + box.h), x1 : int(box.x + box.w)]
            if crop.size:
                crops[track_id] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if len(crops) == len(job.analysis.tracks):
            break
    return crops


with st.sidebar:
    st.header("Input")
    video = st.file_uploader("Video or image", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png"])
    refs = st.file_uploader("Reference photos (one face each)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    threshold = st.slider("Match threshold (cosine)", 0.30, 0.70, PolicyConfig.threshold, 0.01)
    style = st.selectbox("Style", list(Style), index=0)
    keep_audio = st.checkbox("Keep audio", value=False, help="Voices of third parties are personal data too.")
    analyse = st.button("Analyse", type="primary", disabled=video is None)

if analyse and video is not None:
    source = save(video, f"input{Path(video.name).suffix}")
    references = [save(r, f"ref{i}{Path(r.name).suffix}") for i, r in enumerate(refs or [])]
    config = RedactionConfig(
        policy=PolicyConfig(threshold=threshold, frame_threshold=min(PolicyConfig.frame_threshold, threshold)),
        style=style,
        keep_audio=keep_audio,
    )
    with st.spinner("Detecting, embedding and tracking faces..."):
        job = prepare(source, references, config)
        run_analysis(job)
        st.session_state.job = job
        st.session_state.thumbs = thumbnails(job)
        st.session_state.pop("record", None)

job = st.session_state.get("job")
if job is None:
    st.info("Upload a video and reference photos, then press **Analyse**. Without references, everyone is blurred.")
    st.stop()

analysis = job.analysis
st.subheader(
    f"{len(analysis.tracks)} people tracked · {analysis.faces_seen} face detections · {analysis.frame_count} frames"
)

overrides: dict[int, Decision] = {}
columns = st.columns(4)
for i, track in enumerate(analysis.tracks):
    auto = analysis.decisions[track.id]
    with columns[i % 4]:
        if track.id in st.session_state.thumbs:
            st.image(st.session_state.thumbs[track.id], width=140)
        st.markdown(f"**Track {track.id}** · frames {track.first_frame} to {track.last_frame}")
        st.caption(f"{auto.reason}" + (f" · median sim {auto.median_similarity}" if auto.median_similarity else ""))
        choice = st.radio(
            "Decision",
            [Decision.BLUR, Decision.KEEP],
            index=0 if auto.decision is Decision.BLUR else 1,
            key=f"decision-{track.id}",
            horizontal=True,
        )
        overrides[track.id] = choice

if st.button("Render redacted copy", type="primary"):
    apply_overrides(job, overrides)
    output = workdir() / f"redacted{'.png' if job.info.is_image else '.mp4'}"
    with st.spinner("Rendering..."):
        st.session_state.record = render_file(job, output)
        st.session_state.output = output

if record := st.session_state.get("record"):
    output = st.session_state.output
    st.success(
        f"{record['summary']['tracks_kept']} kept, {record['summary']['tracks_blurred']} blurred, "
        f"{record['summary']['manual_overrides']} manual overrides."
    )
    (st.image if job.info.is_image else st.video)(output.read_bytes())
    st.download_button("Download redacted file", output.read_bytes(), file_name=output.name)
    st.download_button(
        "Download audit record (JSON)", json.dumps(record, indent=2), file_name=output.name + ".audit.json"
    )
