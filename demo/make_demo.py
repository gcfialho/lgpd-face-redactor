"""Builds the README before/after GIF from public-domain NASA footage.

1. Download NASA footage of a crew arriving at Kennedy Space Center (public domain, NASA Image and Video Library).
2. Find a ~6 s window where at least three faces are visible the whole time.
3. Pick one crew member as the "data subject" and take their reference photo from a
   different moment of the video (at least 30 s away), so the demo exercises real
   matching rather than comparing a frame with itself.
4. Run the redactor on the clip and render a side-by-side GIF.

    uv run python demo/make_demo.py --out docs/img
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from lgpd_face_redactor.matching import normalize
from lgpd_face_redactor.pipeline import RedactionConfig, redact_file
from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector

NASA_ID = "KSC-20250726-VP-FMX01-0001-SpaceX_Crew_11_Astronaut_Arrival_LLF_720p-M14840"
URL = f"https://images-assets.nasa.gov/video/{NASA_ID}/{NASA_ID}~medium.mp4"
CLIP_SECONDS = 6
MIN_FACES = 2
MIN_FACE_MOTION = 4.0  # px the faces themselves must move between samples; rules out title cards and slides
MIN_SCENE_CHANGE = 12.0  # the reference must come from a visibly different shot than the clip


def thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def face_motion(previous, current) -> float:
    """Mean distance from each face to the nearest face in the previous sample."""
    if not previous or not current:
        return 0.0
    centres = [(f.box.x + f.box.w / 2, f.box.y + f.box.h / 2) for f in previous]
    moves = [
        min(np.hypot(f.box.x + f.box.w / 2 - cx, f.box.y + f.box.h / 2 - cy) for cx, cy in centres) for f in current
    ]
    return float(np.mean(moves))


def sample(video: Path, detector, embedder, step_s: float, max_s: float):
    """Detections (and embeddings) every `step_s` seconds."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    samples = []
    t = 0.0
    while t < max_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        faces = [f for f in detector.detect(frame) if f.box.w >= 40]
        embs = [embedder.embed(frame, f) for f in faces]
        samples.append((t, frame, faces, embs, thumb(frame)))
        t += step_s
    cap.release()
    return samples, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    parser.add_argument("--work", type=Path, default=Path(".demo"))
    parser.add_argument("--scan-seconds", type=float, default=900)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    source = args.work / "source.mp4"
    if not source.exists():
        urllib.request.urlretrieve(URL, source)

    detector, embedder = YuNetDetector(score_threshold=0.8), SFaceEmbedder()
    samples, fps = sample(source, detector, embedder, 1.0, args.scan_seconds)

    # 1. Window of live footage (not a static card) with >= MIN_FACES faces in every sampled second.
    start = None
    for i in range(1, len(samples) - CLIP_SECONDS):
        window = samples[i : i + CLIP_SECONDS + 1]
        if not all(len(s[2]) >= MIN_FACES for s in window):
            continue
        motion = np.mean([face_motion(window[k - 1][2], window[k][2]) for k in range(1, len(window))])
        if motion >= MIN_FACE_MOTION:
            start, start_thumbs = window[0][0], [s[4] for s in window]
            break
    if start is None:
        raise SystemExit(f"No moving {CLIP_SECONDS}s window with {MIN_FACES}+ faces in the first {args.scan_seconds}s")

    # 2. Subject: the second face from the left at the start of the window.
    t0 = next(s for s in samples if s[0] == start)
    order = sorted(range(len(t0[2])), key=lambda j: t0[2][j].box.x)
    candidates = [j for j in order[1:] + order[:1] if t0[3][j] is not None]  # prefer the second from the left
    if not candidates:
        raise SystemExit("No alignable face at the start of the window")
    subject_emb = normalize(t0[3][candidates[0]])

    # 3. Reference photo of the same person from >= 30 s away and a visibly different shot.
    best = None
    for t, frame, faces, embs, th in samples:
        if abs(t - start) < 30 or min(diff(th, w) for w in start_thumbs) < MIN_SCENE_CHANGE:
            continue
        for face, emb in zip(faces, embs, strict=True):
            if emb is None:
                continue
            sim = float(normalize(emb) @ subject_emb)
            if best is None or sim > best[0]:
                best = (sim, t, frame, face)
    if best is None or best[0] < 0.6:
        raise SystemExit("Could not find a reliable reference frame of the subject elsewhere in the video")
    ref_sim, ref_t, ref_frame, ref_face = best
    box = ref_face.box.expanded(1.6)
    x1, y1 = max(0, int(box.x)), max(0, int(box.y))
    reference = ref_frame[y1 : int(box.y + box.h), x1 : int(box.x + box.w)]
    ref_path = args.work / "subject.jpg"
    cv2.imwrite(str(ref_path), reference)
    cv2.imwrite(
        str(args.out / "demo-reference.jpg"),
        cv2.resize(reference, (160, int(160 * reference.shape[0] / reference.shape[1]))),
    )

    # 4. Cut the clip and redact it.
    clip = args.work / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source),
                    "-t", str(CLIP_SECONDS), "-an", "-vf", "scale=640:-2", str(clip)], check=True)  # fmt: skip
    redacted = args.work / "redacted.mp4"
    _, record = redact_file(clip, redacted, [ref_path], RedactionConfig())

    # 5. Side-by-side frames with labels, then a palette-optimised GIF.
    side = args.work / "side.mp4"
    a, b = cv2.VideoCapture(str(clip)), cv2.VideoCapture(str(redacted))
    w, h = int(a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(side), cv2.VideoWriter_fourcc(*"mp4v"), a.get(cv2.CAP_PROP_FPS) or fps, (w * 2 + 8, h + 36)
    )
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        canvas = np.full((h + 36, w * 2 + 8, 3), 255, np.uint8)
        canvas[36:, :w], canvas[36:, w + 8 :] = f1, f2
        for text, x in (("Original", 8), ("lgpd-redact --keep subject.jpg", w + 16)):
            cv2.putText(canvas, text, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()

    gif = args.out / "demo.gif"
    palette = args.work / "palette.png"
    vf = "fps=8,scale=800:-1:flags=lanczos"  # keeps the README GIF around 2-3 MB
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(side), "-vf", f"{vf},palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True)  # fmt: skip
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(side), "-i", str(palette),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5", str(gif)], check=True)  # fmt: skip

    # Small contact sheet (one frame per second) to review the result without the GIF.
    sheet = args.work / "contact-sheet.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(side), "-vf",
                    "fps=1,scale=480:-1,tile=2x3", "-frames:v", "1", str(sheet)], check=True)  # fmt: skip

    meta = {
        "source": f"NASA Image and Video Library, {NASA_ID} (public domain)",
        "clip_start_s": start,
        "reference_from_s": ref_t,
        "reference_similarity_to_subject": round(ref_sim, 3),
        "summary": record["summary"],
        "tracks": record["tracks"],
        "gif_bytes": gif.stat().st_size,
    }
    (args.out / "demo.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
