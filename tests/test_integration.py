"""OpenCV-dependent tests: rendering, media round trips, and the real models wired up correctly."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from conftest import ScriptedDetector, walk  # noqa: E402

from lgpd_face_redactor.cli import main  # noqa: E402
from lgpd_face_redactor.pipeline import RedactionConfig, redact_file  # noqa: E402
from lgpd_face_redactor.render import obscure  # noqa: E402
from lgpd_face_redactor.types import Box, Style  # noqa: E402


def textured(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("style", list(Style))
def test_obscure_changes_only_the_box(style):
    frame = textured()
    out = obscure(frame, [Box(40, 30, 50, 50)], style)
    assert not np.array_equal(out[30:80, 40:90], frame[30:80, 40:90])
    mask = np.ones(frame.shape[:2], bool)
    mask[30:80, 40:90] = False
    assert np.array_equal(out[mask], frame[mask])


def test_obscure_clips_boxes_outside_the_frame():
    frame = textured()
    out = obscure(frame, [Box(-30, -30, 60, 60), Box(500, 500, 10, 10)], Style.BOX)
    assert (out[:30, :30] == 0).all()


def write_video(path, n=20, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n):
        writer.write(textured(size[1], size[0], seed=i))
    writer.release()


def test_blur_all_video_round_trip(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    write_video(source)
    faces = walk(Box(40, 30, 40, 40), 1, 20)
    detector = ScriptedDetector([[(b, 0)] for b in faces])

    _, record = redact_file(source, output, [], RedactionConfig(), detector=detector)

    assert output.exists()
    assert record["summary"]["tracks"] == 1
    assert record["summary"]["tracks_blurred"] == 1
    assert record["settings"]["mode"] == "blur-all"
    assert record["input"]["frames"] == 20
    assert "embedding" not in json.dumps(record)  # no biometric data in the audit trail


def test_image_input(tmp_path):
    source, output = tmp_path / "in.png", tmp_path / "out.png"
    cv2.imwrite(str(source), textured())
    detector = ScriptedDetector([[(Box(40, 30, 40, 40), 0)]])
    redact_file(source, output, [], RedactionConfig(), detector=detector)
    assert not np.array_equal(cv2.imread(str(output)), cv2.imread(str(source)))


@pytest.mark.models
def test_real_models_load_and_run(tmp_path):
    """Downloads YuNet and SFace (verified by SHA-256) and checks the OpenCV API wiring."""
    from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector

    detector = YuNetDetector()
    assert detector.detect(np.zeros((240, 320, 3), np.uint8)) == []
    SFaceEmbedder()


@pytest.mark.models
def test_cli_blur_all_on_image_without_faces(tmp_path, capsys):
    source, output = tmp_path / "in.png", tmp_path / "out.png"
    cv2.imwrite(str(source), np.zeros((120, 160, 3), np.uint8))
    assert main([str(source), "-o", str(output)]) == 0
    record = json.loads((tmp_path / "out.png.audit.json").read_text())
    assert record["summary"]["faces_detected"] == 0
    assert "0 faces" in capsys.readouterr().out
