"""Web app end to end, with scripted faces: upload, gallery, select, render, download, delete."""

import io

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("fastapi")

from conftest import IdentityEmbedder, ScriptedDetector  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lgpd_face_redactor.types import Box  # noqa: E402
from lgpd_face_redactor.web.app import create_app  # noqa: E402
from lgpd_face_redactor.web.jobs import JobStore, Status  # noqa: E402

ALICE, BOB = 1, 2
LEFT, RIGHT = Box(20, 30, 50, 50), Box(140, 30, 50, 50)


def png() -> bytes:
    image = np.random.default_rng(0).integers(0, 255, (120, 220, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_path):
    detector = ScriptedDetector([[(LEFT, ALICE), (RIGHT, BOB)]] * 10)
    store = JobStore(lambda: detector, IdentityEmbedder, root=tmp_path, inline=True)
    with TestClient(create_app(store)) as c:
        c.store = store
        yield c


def upload(client) -> str:
    response = client.post(
        "/jobs", files={"file": ("party.png", io.BytesIO(png()), "image/png")}, data={"style": "box"}
    )
    assert response.status_code == 200  # followed the 303 to the job page
    return str(response.url).rstrip("/").split("/")[-1]


def test_upload_shows_one_card_per_person(client):
    job_id = upload(client)
    job = client.store.get(job_id)
    assert job.status is Status.READY
    assert len(job.people) == 2

    page = client.get(f"/jobs/{job_id}").text
    assert page.count('name="keep"') == 2
    assert client.get(f"/jobs/{job_id}/people/0.jpg").headers["content-type"] == "image/jpeg"


def test_keeping_one_person_blurs_the_other(client):
    job_id = upload(client)
    job = client.store.get(job_id)
    alice = next(p for p in job.people if p.best.box.x == LEFT.x)

    client.post(f"/jobs/{job_id}/render", data={"keep": [alice.id], "style": "box"})

    assert job.status is Status.DONE
    out = cv2.imdecode(np.frombuffer(client.get(f"/jobs/{job_id}/download").content, np.uint8), cv2.IMREAD_COLOR)
    original = cv2.imdecode(np.frombuffer(png(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out[40:70, 30:60], original[40:70, 30:60])  # Alice untouched
    assert (out[40:70, 150:180] == 0).all()  # Bob boxed out

    record = client.get(f"/jobs/{job_id}/audit.json").json()
    assert record["settings"]["mode"] == "selected-people"
    assert record["settings"]["people_kept_visible"] == 1
    assert record["input"]["file"] == "party.png"


def test_rendering_with_nobody_selected_blurs_everyone(client):
    job_id = upload(client)
    client.post(f"/jobs/{job_id}/render", data={"style": "box"})
    assert client.get(f"/jobs/{job_id}/audit.json").json()["summary"]["tracks_blurred"] == 2


def test_delete_removes_every_file(client):
    job_id = upload(client)
    folder = client.store.get(job_id).folder
    assert folder.exists()
    client.post(f"/jobs/{job_id}/delete")
    assert not folder.exists()
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_rejects_unsupported_files(client):
    response = client.post("/jobs", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_privacy_headers(client):
    headers = client.get("/").headers
    assert headers["cache-control"] == "no-store"
    assert "default-src 'self'" in headers["content-security-policy"]
