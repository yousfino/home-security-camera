from types import SimpleNamespace
import numpy as np
import pytest
from security_ai.config import RecognitionConfig
from security_ai.core import GalleryIndex
from security_ai.detection import PersonDetection
from security_ai.tracking import PersonTracker
from security_ai.pipeline import RecognitionPipeline
from security_ai.cli import enrollment_sample
from security_ai.storage import load_gallery, save_gallery


class Detector:
    def detect(self, frame):
        return [PersonDetection((0, 0, 100, 200), 0.9), PersonDetection((200, 0, 300, 200), 0.9)]


class Analyzer:
    good = True

    def detect(self, frame):
        return [
            SimpleNamespace(bbox=np.array([10, 10, 50, 60]), embedding=np.array([1.0, 0.0])),
            SimpleNamespace(bbox=np.array([210, 10, 250, 60]), embedding=np.array([0.0, 1.0])),
        ]

    def embed(self, frame, face):
        return face.embedding


def pipeline(analyzer=None, **config):
    return RecognitionPipeline(
        Detector(),
        PersonTracker(),
        analyzer or Analyzer(),
        {"alice": [np.array([1.0, 0.0])]},
        RecognitionConfig(**{"recognition_interval": 0, **config}),
        quality=lambda f, face: f[0, 0, 0] == 0,
    )


def test_independent_people_decisions_and_cooldowns():
    p = pipeline()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for t in (0, 0.1, 0.2):
        results = p.process(frame, t)
    assert [r.status for r in results] == ["familiar", "unfamiliar"]
    assert all(r.emit for r in results)
    p.acknowledge(results[0], 0.2)
    results = p.process(frame, 0.3)
    assert not results[0].emit and results[1].emit
    p.acknowledge(results[1], 0.3)
    assert not any(r.emit for r in p.process(frame, 0.4))
    assert all(r.status == "unresolved" and not r.emit for r in p.process(frame + 1, 0.5))


def test_no_face_never_reuses_identity_and_ttl_resets():
    analyzer = Analyzer()
    p = pipeline(analyzer, evidence_ttl=0.2)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for t in (0, 0.1, 0.2):
        p.process(frame, t)
    assert all(r.status == "unresolved" for r in p.process(frame, 0.5))
    analyzer.detect = lambda f: []
    assert all(r.status == "unresolved" for r in p.process(frame, 0.6))


def test_gallery_dimensions_ambiguity_and_enrollment(tmp_path):
    with pytest.raises(ValueError):
        GalleryIndex({"a": [np.zeros(2)]})
    with pytest.raises(ValueError):
        GalleryIndex({"a": [np.ones(2), np.ones(3)]})
    p = pipeline()
    p.gallery = GalleryIndex({"a": [np.array([1.0, 0.0])], "b": [np.array([1.0, 0.01])]})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for t in (0, 0.1, 0.2):
        results = p.process(frame, t)
    assert results[0].status == "unresolved"
    with pytest.raises(ValueError, match="Duplicate"):
        enrollment_sample([np.array([1.0, 0.0])], np.array([1.0, 0.0]), 0.6, 0.995)
    with pytest.raises(ValueError, match="differs"):
        enrollment_sample([np.array([1.0, 0.0])], np.array([0.0, 1.0]), 0.6, 0.995)
    path = tmp_path / "gallery.npz"
    save_gallery(path, {"a": [np.ones(2)]}, model_id="v1")
    assert "a" in load_gallery(path, "v1")
    with pytest.raises(ValueError, match="model mismatch"):
        load_gallery(path, "v2")


def test_empty_gallery_is_unresolved_not_unfamiliar():
    p = pipeline(minimum=1)
    p.gallery = GalleryIndex({})
    results = p.process(np.zeros((240, 320, 3), np.uint8), 0)
    assert all(r.status == "unresolved" and not r.emit for r in results)


def test_gallery_rejects_matrix_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        GalleryIndex({"alice": [np.ones((2, 2))]})
