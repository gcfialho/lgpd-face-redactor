"""Detector recall on the WIDER FACE validation set, by face size.

Every missed face is a potential leak, so recall (not precision) is the metric that
matters for redaction. A ground-truth face counts as found if any detection overlaps it
with IoU >= 0.5, or covers most of it once expanded the way the redactor expands boxes.

    uv run python eval/wider_recall.py --data .eval-data --out eval/results/wider.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2

from lgpd_face_redactor.policy import PolicyConfig
from lgpd_face_redactor.types import Box
from lgpd_face_redactor.vision import YuNetDetector

BASE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
FILES = {"WIDER_val.zip": "WIDER_val", "wider_face_split.zip": "wider_face_split"}
BUCKETS = [
    (0, 16, "tiny (<16 px)"),
    (16, 32, "small (16-32 px)"),
    (32, 64, "medium (32-64 px)"),
    (64, 10_000, "large (>=64 px)"),
]


def download(data: Path) -> Path:
    data.mkdir(parents=True, exist_ok=True)
    for name, folder in FILES.items():
        if not (data / folder).exists():
            archive = data / name
            if not archive.exists():
                urllib.request.urlretrieve(f"{BASE}/{name}", archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(data)
    return data


def read_annotations(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        name, count = lines[i], int(lines[i + 1])
        rows = lines[i + 2 : i + 2 + max(count, 1)]
        boxes = []
        for row in rows[:count]:
            # x y w h blur expression illumination invalid occlusion pose
            x, y, w, h, _blur, _expr, _illum, invalid, _occl, _pose = map(int, row.split()[:10])
            if not invalid and w > 0 and h > 0:
                boxes.append(Box(x, y, w, h))
        yield name, boxes
        i += 2 + max(count, 1)


def covered(gt: Box, detections: list[Box], expansion: float) -> bool:
    for d in detections:
        if gt.iou(d) >= 0.5:
            return True
        e = d.expanded(expansion)
        ix = max(0.0, min(gt.x + gt.w, e.x + e.w) - max(gt.x, e.x))
        iy = max(0.0, min(gt.y + gt.h, e.y + e.h) - max(gt.y, e.y))
        if gt.area and ix * iy / gt.area >= 0.8:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".eval-data"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/wider.json"))
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images (0 = all)")
    args = parser.parse_args()

    data = download(args.data)
    detector = YuNetDetector()
    expansion = PolicyConfig().box_expansion
    found = {label: 0 for *_, label in BUCKETS}
    total = dict(found)
    images = 0

    annotations = data / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    for name, gts in read_annotations(annotations):
        if args.limit and images >= args.limit:
            break
        image = cv2.imread(str(data / "WIDER_val" / "images" / name))
        if image is None:
            continue
        images += 1
        detections = [d.box for d in detector.detect(image)]
        for gt in gts:
            size = min(gt.w, gt.h)
            label = next(lbl for lo, hi, lbl in BUCKETS if lo <= size < hi)
            total[label] += 1
            found[label] += covered(gt, detections, expansion)

    result = {
        "dataset": "WIDER FACE validation",
        "images": images,
        "detector_score_threshold": detector.score_threshold,
        "recall_by_size": {
            k: {"faces": total[k], "recall": round(found[k] / total[k], 4) if total[k] else None} for k in total
        },
        "recall_overall": round(sum(found.values()) / max(1, sum(total.values())), 4),
        "recall_faces_16px_and_up": round(
            sum(found[k] for k in total if not k.startswith("tiny"))
            / max(1, sum(total[k] for k in total if not k.startswith("tiny"))),
            4,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
