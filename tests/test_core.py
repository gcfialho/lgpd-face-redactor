import numpy as np
import pytest
from conftest import identity_vector

from lgpd_face_redactor.matching import ReferenceGallery, normalize
from lgpd_face_redactor.policy import Decision, PolicyConfig, decide, plan_render
from lgpd_face_redactor.tracking import IouTracker
from lgpd_face_redactor.types import Box, Detection, Observation, Track


class TestBox:
    def test_iou(self) -> None:
        a = Box(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(Box(5, 0, 10, 10)) == pytest.approx(50 / 150)
        assert a.iou(Box(20, 20, 5, 5)) == 0.0

    def test_expanded_keeps_centre(self) -> None:
        e = Box(10, 10, 20, 40).expanded(1.5)
        assert (e.x + e.w / 2, e.y + e.h / 2) == (20, 30)
        assert (e.w, e.h) == (30, 60)

    def test_lerp(self) -> None:
        assert Box.lerp(Box(0, 0, 10, 10), Box(10, 0, 20, 10), 0.5) == Box(5, 0, 15, 10)


class TestGallery:
    def test_best_similarity_is_max_over_references(self) -> None:
        gallery = ReferenceGallery([identity_vector(1), identity_vector(2)])
        assert gallery.best_similarity(identity_vector(2)) == pytest.approx(1.0, abs=1e-5)
        assert gallery.best_similarity(identity_vector(3)) < 0.4

    def test_rejects_mismatched_dimensions_and_zero_vectors(self) -> None:
        gallery = ReferenceGallery([np.ones(4)])
        with pytest.raises(ValueError):
            gallery.add(np.ones(5))
        with pytest.raises(ValueError):
            normalize(np.zeros(4))

    def test_empty_gallery_cannot_match(self) -> None:
        with pytest.raises(ValueError):
            ReferenceGallery().best_similarity(np.ones(4))


def det(box: Box) -> Detection:
    return Detection(box, 0.9)


class TestTracker:
    def test_follows_two_people_moving_apart(self) -> None:
        tracker = IouTracker()
        for f in range(10):
            boxes = [Box(10 + f, 10, 40, 40), Box(200 - f, 10, 40, 40)]
            tracker.update(f, [det(b) for b in boxes], [None, None])
        tracks = tracker.finish()
        assert [len(t.observations) for t in tracks] == [10, 10]

    def test_appearance_gate_splits_overlapping_different_people(self) -> None:
        tracker = IouTracker()
        box = Box(10, 10, 40, 40)
        tracker.update(0, [det(box)], [identity_vector(1)])
        tracker.update(1, [det(box)], [identity_vector(2)])  # same place, different face
        assert len(tracker.finish()) == 2

    def test_stores_normalised_embeddings(self) -> None:
        tracker = IouTracker()
        tracker.update(0, [det(Box(10, 10, 40, 40))], [identity_vector(1) * 7])
        (track,) = tracker.finish()
        embedding = track.observations[0].embedding
        assert embedding is not None
        assert np.linalg.norm(embedding.astype(np.float32)) == pytest.approx(1, abs=1e-3)

    def test_closes_track_after_long_gap(self) -> None:
        tracker = IouTracker()
        box = Box(10, 10, 40, 40)
        tracker.update(0, [det(box)], [None])
        tracker.update(50, [det(box)], [None])
        assert len(tracker.finish()) == 2


def track_with(similarities: list[float | None], start: int = 0) -> Track:
    obs = [Observation(start + i, Box(10, 10, 40, 40), 0.9, s) for i, s in enumerate(similarities)]
    return Track(id=0, observations=obs)


class TestPolicy:
    config = PolicyConfig(threshold=0.42, frame_threshold=0.36, min_frames=5, min_matched_share=0.6, pad_frames=2)

    def test_keeps_consistent_match(self) -> None:
        assert decide(track_with([0.6] * 10), self.config, True).decision is Decision.KEEP

    @pytest.mark.parametrize(
        ("sims", "reason"),
        [
            ([0.6] * 3, "too short"),
            ([0.3] * 10, "median similarity"),
            ([0.6] * 5 + [0.1] * 4 + [None], "frames match"),  # median passes; share fails, None counts against
            ([None] * 10, "could not be compared"),
        ],
    )
    def test_blurs_when_evidence_is_weak(self, sims: list[float | None], reason: str) -> None:
        result = decide(track_with(sims), self.config, True)
        assert result.decision is Decision.BLUR
        assert reason in result.reason

    def test_no_reference_blurs_everyone(self) -> None:
        assert decide(track_with([0.9] * 10), self.config, False).decision is Decision.BLUR

    def test_kept_track_still_blurs_its_weak_frames(self) -> None:
        track = track_with([0.6] * 4 + [0.2] + [0.6] * 4 + [None])
        decision = {0: decide(track, self.config, True)}
        plan = plan_render([track], decision, self.config, frame_count=10)
        assert decision[0].decision is Decision.KEEP
        assert sorted(plan.blur_boxes) == [4, 9]

    def test_blurred_track_fills_gaps_and_pads_edges(self) -> None:
        track = Track(0, [Observation(5, Box(0, 0, 10, 10), 0.9, 0.1), Observation(9, Box(40, 0, 10, 10), 0.9, 0.1)])
        decision = {0: decide(track, self.config, True)}
        plan = plan_render([track], decision, self.config, frame_count=12)
        # 3..4 padding, 5..9 observed + interpolated, 10..11 padding
        assert sorted(plan.blur_boxes) == list(range(3, 12))
        middle = plan.blur_boxes[7][0]
        assert middle.x + middle.w / 2 == pytest.approx(25)  # halfway between the two sightings
