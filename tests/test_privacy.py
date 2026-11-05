import sqlite3
import numpy as np
import pytest
from security_ai.storage import (
    data_paths,
    init_db,
    save_profile,
    save_event,
    delete_profile,
    load_gallery,
    StorageLimitError,
    drop_crops,
)
from security_ai import cli


def test_profile_deletion_purges_linked_episode_and_preserves_other_profiles(tmp_path):
    gallery, db, crops = data_paths(tmp_path / "private")
    save_profile(gallery, "alice", [np.ones(2)], "model")
    save_profile(gallery, "bob", [np.array([1.0, 0.0])], "model")
    init_db(db)
    save_event(db, crops, "familiar", "alice", 0.9, None, episode_id="e1")
    save_event(db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8), episode_id="e1")
    save_event(db, crops, "familiar", "bob", 0.9, None, episode_id="e2")
    result = delete_profile(gallery, db, crops, "alice")
    assert result["profile_removed"] and result["events_removed"] == 2
    assert set(load_gallery(gallery, "model")) == {"bob"}
    assert list(crops.glob("*.jpg")) == []
    assert not delete_profile(gallery, db, crops, "alice")["profile_removed"]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT profile_name FROM events").fetchall() == [("bob",)]


def test_budget_rejects_new_crop_and_cleans_staging(tmp_path):
    _, db, crops = data_paths(tmp_path / "private")
    init_db(db)
    with pytest.raises(StorageLimitError):
        save_event(
            db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8), max_storage_bytes=1
        )
    assert not list(crops.iterdir())
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0


def test_crop_removal_and_redacted_listing(tmp_path, capsys):
    _, db, crops = data_paths(tmp_path / "private")
    init_db(db)
    save_event(db, crops, "familiar", "alice", 0.9, None)
    save_event(db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8))
    cli.list_events(db)
    assert "alice" not in capsys.readouterr().out
    cli.list_events(db, True)
    assert "alice" in capsys.readouterr().out
    drop_crops(db, crops)
    assert not list(crops.glob("*.jpg"))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert (
            conn.execute("SELECT count(*) FROM events WHERE image_path IS NOT NULL").fetchone()[0]
            == 0
        )


def test_deletion_cli_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["security-ai", "profile-delete", "--name", "alice"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
