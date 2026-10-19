"""Groups tracks into people, so a reviewer picks a person once instead of every track.

The grouping is deliberately conservative. Wrongly merging two people would let a single
click reveal a stranger, while wrongly splitting one person only means two cards to tick.
So we use average-linkage clustering with a merge threshold above the verification point,
and never merge tracks that appear in the same frame: one person cannot be in two places.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lgpd_face_redactor.types import Observation, Track


@dataclass
class Person:
    id: int
    track_ids: list[int]
    frames: int
    first_frame: int
    last_frame: int
    best: Observation  # most confident sighting, used for the thumbnail
    embeddings: list[np.ndarray] = field(default_factory=list, repr=False)  # one mean embedding per track

    @property
    def comparable(self) -> bool:
        """False when no face of this person could be embedded; such people can only be blurred."""
        return bool(self.embeddings)


def _similarity(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return float(np.mean([x @ y for x in a for y in b]))


def group_people(tracks: list[Track], merge_threshold: float = 0.5) -> list[Person]:
    clusters: list[dict] = []
    for track in tracks:
        mean = track.mean_embedding()
        clusters.append({"tracks": [track], "embeddings": [mean] if mean is not None else [], "frames": track.frames()})

    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if not a["embeddings"] or not b["embeddings"] or a["frames"] & b["frames"]:
                    continue
                sim = _similarity(a["embeddings"], b["embeddings"])
                if sim >= merge_threshold and (best is None or sim > best[0]):
                    best = (sim, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = clusters[i], clusters.pop(j)
        a["tracks"] += b["tracks"]
        a["embeddings"] += b["embeddings"]
        a["frames"] |= b["frames"]

    people = []
    for cluster in clusters:
        observations = [o for t in cluster["tracks"] for o in t.observations]
        people.append(
            Person(
                id=0,
                track_ids=sorted(t.id for t in cluster["tracks"]),
                frames=len(cluster["frames"]),
                first_frame=min(o.frame for o in observations),
                last_frame=max(o.frame for o in observations),
                best=max(observations, key=lambda o: o.score * o.box.area),
                embeddings=cluster["embeddings"],
            )
        )
    people.sort(key=lambda p: (-p.frames, p.first_frame))
    for index, person in enumerate(people):
        person.id = index
    return people


def gallery_embeddings(people: list[Person], selected: set[int]) -> list[np.ndarray]:
    """Reference embeddings for the people to keep: one per track, so pose variety is preserved."""
    return [e for p in people if p.id in selected for e in p.embeddings]
