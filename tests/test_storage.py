import numpy as np
from security_ai.storage import save_gallery, load_gallery, safe_profile_name, init_db, save_event


def test_gallery_roundtrip(tmp_path):
    p = tmp_path / "gallery.npz"
    save_gallery(p, {"alice": [np.array([1.0, 0.0])]})
    assert np.allclose(load_gallery(p)["alice"][0], [1.0, 0.0])


def test_profile_validation():
    assert safe_profile_name("alice_1") == "alice_1"
    try:
        safe_profile_name("../bad")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid name")


def test_event(tmp_path):
    db = tmp_path / "events.db"
    init_db(db)
    assert save_event(db, tmp_path / "crops", "familiar", "alice", 0.9, None) == 1
