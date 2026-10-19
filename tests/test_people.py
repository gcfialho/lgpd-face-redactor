"""Grouping tracks into people for the review gallery, and keeping the chosen ones."""

from conftest import IdentityEmbedder, ScriptedDetector, identity_vector, walk

from lgpd_face_redactor.matching import ReferenceGallery
from lgpd_face_redactor.people import gallery_embeddings, group_people
from lgpd_face_redactor.pipeline import RedactionConfig, analyze, assess
from lgpd_face_redactor.policy import Decision
from lgpd_face_redactor.types import Box

ALICE, BOB, CAROL = 1, 2, 3


def analysed(script, blank_frames):
    config = RedactionConfig()
    return analyze(blank_frames(len(script)), ScriptedDetector(script), IdentityEmbedder(), config), config


def test_person_who_leaves_and_returns_is_one_card(blank_frames):
    alice_first, alice_back = walk(Box(20, 50, 40, 40), 1, 10), walk(Box(220, 50, 40, 40), -1, 10)
    script = [[(alice_first[f], ALICE)] for f in range(10)] + [[]] * 30 + [[(alice_back[f], ALICE)] for f in range(10)]
    analysis, _ = analysed(script, blank_frames)

    assert len(analysis.tracks) == 2  # the gap closed the first track
    (person,) = group_people(analysis.tracks)
    assert person.track_ids == [t.id for t in analysis.tracks]
    assert person.frames == 20


def test_different_people_are_separate_cards(blank_frames):
    a, b = walk(Box(20, 50, 40, 40), 1, 15), walk(Box(200, 50, 40, 40), -1, 15)
    analysis, _ = analysed([[(a[f], ALICE), (b[f], BOB)] for f in range(15)], blank_frames)
    assert len(group_people(analysis.tracks)) == 2


def test_tracks_seen_in_the_same_frame_are_never_merged(blank_frames):
    """Even if two faces looked alike, one person cannot be in two places at once."""
    a, b = walk(Box(20, 50, 40, 40), 1, 15), walk(Box(200, 50, 40, 40), -1, 15)
    analysis, _ = analysed([[(a[f], ALICE), (b[f], ALICE)] for f in range(15)], blank_frames)
    assert len(group_people(analysis.tracks)) == 2


def test_unembeddable_faces_form_people_that_cannot_be_kept(blank_frames):
    analysis, _ = analysed([[(Box(0, 50, 40, 40), -1)] for _ in range(10)], blank_frames)
    (person,) = group_people(analysis.tracks)
    assert not person.comparable


def test_people_are_ordered_by_screen_time(blank_frames):
    a, b = walk(Box(20, 50, 40, 40), 1, 30), walk(Box(200, 50, 40, 40), -1, 30)
    script = [[(a[f], ALICE)] + ([(b[f], BOB)] if f < 10 else []) for f in range(30)]
    analysis, _ = analysed(script, blank_frames)
    people = group_people(analysis.tracks)
    assert [p.frames for p in people] == [30, 10]
    assert [p.id for p in people] == [0, 1]


def test_selecting_a_person_keeps_only_them(blank_frames):
    a, b, c = (walk(Box(x, 50, 40, 40), 0, 20) for x in (10, 120, 230))
    script = [[(a[f], ALICE), (b[f], BOB), (c[f], CAROL)] for f in range(20)]
    analysis, config = analysed(script, blank_frames)
    people = group_people(analysis.tracks)
    bob = next(p for p in people if p.best.box.x == 120)

    assess(analysis, ReferenceGallery(gallery_embeddings(people, {bob.id})), config)

    kept = {tid for tid, d in analysis.decisions.items() if d.decision is Decision.KEEP}
    assert kept == set(bob.track_ids)


def test_reference_photo_matches_the_right_person(blank_frames):
    """The web app pre-selects people who match an uploaded reference photo."""
    a, b = walk(Box(20, 50, 40, 40), 1, 15), walk(Box(200, 50, 40, 40), -1, 15)
    analysis, _ = analysed([[(a[f], ALICE), (b[f], BOB)] for f in range(15)], blank_frames)
    people = group_people(analysis.tracks)
    reference = ReferenceGallery([identity_vector(BOB)])
    match = max(people, key=lambda p: max(reference.best_similarity(e) for e in p.embeddings))
    assert match.best.box.x > 100  # Bob walks on the right-hand side
