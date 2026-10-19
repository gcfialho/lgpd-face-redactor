import hashlib

import pytest

from lgpd_face_redactor import models


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("LGPD_REDACTOR_MODELS", str(tmp_path))
    payload = b"fake onnx weights"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    return models.ModelSpec("fake", "fake.onnx", source.as_uri(), hashlib.sha256(payload).hexdigest(), "MIT")


def test_downloads_once_and_verifies(spec, tmp_path):
    path = models.ensure(spec)
    assert path == tmp_path / "fake.onnx"
    assert models.sha256_of(path) == spec.sha256
    path.touch()
    assert models.ensure(spec) == path  # cached


def test_rejects_tampered_download(spec):
    bad = models.ModelSpec(spec.name, spec.filename, spec.url, "0" * 64, spec.license)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        models.ensure(bad)
    assert not (models.cache_dir() / "fake.onnx").exists()


def test_pinned_hashes_look_like_sha256():
    for m in (models.YUNET, models.SFACE):
        assert len(m.sha256) == 64
        assert m.url.startswith("https://")
