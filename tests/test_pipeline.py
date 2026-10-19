"""End-to-end analysis with scripted faces: the scenarios a redaction tool must get right."""

from conftest import IdentityEmbedder, ScriptedDetector, identity_vector, walk

from lgpd_face_redactor.matching import ReferenceGallery
from lgpd_face_redactor.pipeline import RedactionConfig, analyze
from lgpd_face_redactor.policy import Decision
from lgpd_face_redactor.types import Box

SUBJECT, STRANGER, OTHER = 1, 2, 3


def run(script, blank_frames, references=(SUBJECT,), noise=0.15):
    gallery = ReferenceGallery([identity_vector(r) for r in references])
    return analyze(
        blank_frames(len(script)), ScriptedDetector(script), IdentityEmbedder(noise), gallery, RedactionConfig()
    )


def visible(analysis, frame: int, box: Box) -> bool:
    """A face is visible if no planned blur box covers its centre."""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    return not any(b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h for b in analysis.plan.blur_boxes.get(frame, []))


def test_subject_visible_stranger_blurred(blank_frames):
    subject, stranger = walk(Box(20, 50, 40, 40), 1, 30), walk(Box(220, 50, 40, 40), -1, 30)
    script = [[(subject[f], SUBJECT), (stranger[f], STRANGER)] for f in range(30)]
    analysis = run(script, blank_frames)

    decisions = sorted(d.decision for d in analysis.decisions.values())
    assert decisions == [Decision.BLUR, Decision.KEEP]
    assert all(visible(analysis, f, subject[f]) for f in range(30))
    assert not any(visible(analysis, f, stranger[f]) for f in range(30))


def test_no_reference_blurs_everyone(blank_frames):
    faces = walk(Box(20, 50, 40, 40), 1, 10)
    analysis = run([[(faces[f], SUBJECT)] for f in range(10)], blank_frames, references=())
    assert not any(visible(analysis, f, faces[f]) for f in range(10))


def test_stranger_missed_for_a_few_frames_stays_blurred(blank_frames):
    """Detector misses (blur, motion, occlusion) must not reveal a third party in between."""
    stranger = walk(Box(100, 50, 40, 40), 2, 20)
    script = [[] if 8 <= f <= 11 else [(stranger[f], STRANGER)] for f in range(20)]
    analysis = run(script, blank_frames)
    assert not any(visible(analysis, f, stranger[f]) for f in range(20))


def test_brief_glimpse_of_subject_is_blurred(blank_frames):
    """Two frames are not enough evidence; fail safe."""
    face = Box(50, 50, 40, 40)
    script = [[(face, SUBJECT)] if f < 2 else [] for f in range(10)]
    analysis = run(script, blank_frames)
    assert not visible(analysis, 0, face)


def test_stranger_stepping_into_subject_position_is_not_revealed(blank_frames):
    """Identity switch: the subject leaves and a stranger appears in the same spot."""
    spot = Box(80, 50, 40, 40)
    script = [[(spot, SUBJECT)] if f < 15 else [(spot, STRANGER)] for f in range(30)]
    analysis = run(script, blank_frames)
    pad = RedactionConfig().policy.pad_frames
    # The stranger's track is padded backwards, so the subject is briefly blurred just before
    # the switch. That over-redaction is the intended price of never under-redacting.
    assert all(visible(analysis, f, spot) for f in range(15 - pad))
    assert not any(visible(analysis, f, spot) for f in range(15, 30))


def test_unalignable_faces_are_blurred(blank_frames):
    edge_face = Box(0, 50, 40, 40)
    script = [[(edge_face, -1)] for _ in range(10)]
    analysis = run(script, blank_frames)
    assert not any(visible(analysis, f, edge_face) for f in range(10))


def test_several_people_can_be_kept(blank_frames):
    a, b = walk(Box(20, 50, 40, 40), 1, 20), walk(Box(200, 50, 40, 40), -1, 20)
    script = [[(a[f], SUBJECT), (b[f], OTHER)] for f in range(20)]
    analysis = run(script, blank_frames, references=(SUBJECT, OTHER))
    assert all(d.decision is Decision.KEEP for d in analysis.decisions.values())


def test_reviewer_override_is_applied_and_explained(blank_frames):
    from lgpd_face_redactor.pipeline import Job, apply_overrides

    subject = walk(Box(20, 50, 40, 40), 1, 20)
    analysis = run([[(subject[f], SUBJECT)] for f in range(20)], blank_frames)
    (track_id,) = analysis.decisions
    assert analysis.decisions[track_id].decision is Decision.KEEP

    job = Job(None, None, RedactionConfig(), None, None, ReferenceGallery(), analysis=analysis)
    updated = apply_overrides(job, {track_id: Decision.BLUR})

    assert updated.decisions[track_id].decision is Decision.BLUR
    assert "manual override" in updated.decisions[track_id].reason
    assert not any(visible(updated, f, subject[f]) for f in range(20))
