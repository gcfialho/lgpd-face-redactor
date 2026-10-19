"""Calibrates the match threshold on LFW (6,000 standard verification pairs, 10 folds).

For redaction the costly error is a false accept (a stranger kept visible), so we report
the threshold needed for a target false accept rate, not just the accuracy-optimal one.

    uv run python eval/lfw_verification.py --data .eval-data --out eval/results/lfw.json
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from lgpd_face_redactor.matching import normalize
from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector

# Same figshare mirrors scikit-learn uses for LFW.
IMAGES_URL = "https://ndownloader.figshare.com/files/5976015"  # lfw-funneled.tgz
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"  # pairs.txt


def download(data: Path) -> tuple[Path, Path]:
    data.mkdir(parents=True, exist_ok=True)
    tgz, pairs = data / "lfw-funneled.tgz", data / "pairs.txt"
    if not tgz.exists():
        urllib.request.urlretrieve(IMAGES_URL, tgz)
    if not pairs.exists():
        urllib.request.urlretrieve(PAIRS_URL, pairs)
    root = data / "lfw_funneled"
    if not root.exists():
        with tarfile.open(tgz) as t:
            t.extractall(data, filter="data")
    return root, pairs


def read_pairs(path: Path) -> list[tuple[str, int, str, int, bool]]:
    lines = path.read_text().split("\n")[1:]
    pairs = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            pairs.append((parts[0], int(parts[1]), parts[0], int(parts[2]), True))
        elif len(parts) == 4:
            pairs.append((parts[0], int(parts[1]), parts[2], int(parts[3]), False))
    return pairs


def central_face(image: np.ndarray, detector: YuNetDetector):
    faces = detector.detect(image)
    if not faces:
        return None
    cy, cx = image.shape[0] / 2, image.shape[1] / 2
    return min(faces, key=lambda d: (d.box.x + d.box.w / 2 - cx) ** 2 + (d.box.y + d.box.h / 2 - cy) ** 2)


def roc(scores: np.ndarray, same: np.ndarray) -> dict:
    order = np.argsort(-scores)
    s, y = scores[order], same[order]
    tpr = np.cumsum(y) / y.sum()
    fpr = np.cumsum(~y) / (~y).sum()
    auc = float(np.trapezoid(np.concatenate([[0], tpr]), np.concatenate([[0], fpr])))

    def at_far(target: float) -> dict:
        ok = np.where(fpr <= target)[0]
        i = ok[-1] if len(ok) else 0
        return {"far": target, "threshold": float(s[i]), "tar": float(tpr[i])}

    return {"auc": auc, "operating_points": [at_far(t) for t in (1e-1, 1e-2, 1e-3)]}


def fold_accuracy(scores: np.ndarray, same: np.ndarray, folds: int = 10) -> dict:
    """Standard LFW protocol: pick the best threshold on 9 folds, test on the 10th."""
    idx = np.arange(len(scores)) % folds
    accs, thresholds = [], []
    candidates = np.linspace(-0.2, 1.0, 601)
    for k in range(folds):
        train, test = idx != k, idx == k
        best = max(candidates, key=lambda t: np.mean((scores[train] >= t) == same[train]))
        thresholds.append(float(best))
        accs.append(float(np.mean((scores[test] >= best) == same[test])))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "threshold_mean": float(np.mean(thresholds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".eval-data"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/lfw.json"))
    args = parser.parse_args()

    root, pairs_file = download(args.data)
    detector, embedder = YuNetDetector(), SFaceEmbedder()
    cache: dict[Path, np.ndarray | None] = {}

    def embedding(name: str, number: int) -> np.ndarray | None:
        path = root / name / f"{name}_{number:04d}.jpg"
        if path not in cache:
            image = cv2.imread(str(path))
            face = central_face(image, detector) if image is not None else None
            vec = embedder.embed(image, face) if face is not None else None
            cache[path] = normalize(vec) if vec is not None else None
        return cache[path]

    scores, same, failed = [], [], 0
    for a, i, b, j, is_same in read_pairs(pairs_file):
        ea, eb = embedding(a, i), embedding(b, j)
        if ea is None or eb is None:
            failed += 1
            # Count failures as non-matches: in production an unmatchable face is blurred.
            scores.append(-1.0)
        else:
            scores.append(float(ea @ eb))
        same.append(is_same)

    scores_np, same_np = np.array(scores), np.array(same)
    result = {
        "dataset": "LFW funneled, 6000 pairs (pairs.txt)",
        "pairs": len(scores),
        "detection_or_alignment_failures": failed,
        **fold_accuracy(scores_np, same_np),
        **roc(scores_np, same_np),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
