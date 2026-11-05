"""Hardware-free end-to-end tests with synthetic faces and mocked inference."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
import os
import sqlite3
from unittest.mock import Mock
import numpy as np
import pytest

from security_ai import cli
from security_ai.capture import CapturedFrame
from security_ai.storage import (
    data_paths,
    init_db,
    save_event,
    maintain_events,
    save_profile,
    load_gallery,
    save_gallery,
)
from test_pipeline import pipeline, Analyzer


def test_migrate_legacy_events_and_private_permissions(tmp_path):
    gallery, db, crops = data_paths(tmp_path / "private")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "captured_at TEXT NOT NULL,status TEXT NOT NULL,profile_name TEXT,"
            "similarity REAL,image_path TEXT)"
        )
        conn.execute("INSERT INTO events(captured_at,status) VALUES ('2020','unresolved')")
    init_db(db)
    event_id = save_event(
        db,
        crops,
        "unfamiliar",
        None,
        0.2,
        np.zeros((8, 8, 3), np.uint8),
        track_id=42,
        session_id="session",
        model_id="model",
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        row = conn.execute(
            "SELECT track_id,session_id,image_path FROM events WHERE id=?", (event_id,)
        ).fetchone()
        assert row[:2] == (42, "session")
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    if os.name != "nt":
        assert db.stat().st_mode & 0o777 == 0o600
        assert Path(row[2]).stat().st_mode & 0o777 == 0o600
        assert db.parent.stat().st_mode & 0o777 == 0o700


def test_image_failure_rolls_back_and_reconciliation(tmp_path, monkeypatch):
    _, db, crops = data_paths(tmp_path / "private")
    init_db(db)
    original = Path.replace

    def failed_replace(self, target):
        if self.name.startswith(".pending-"):
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="disk full"):
        save_event(db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8))
    assert list(crops.iterdir()) == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    (crops / "event_000001.jpg").touch()
    (crops / ".pending-orphan").touch()
    (crops / "user.txt").touch()
    maintain_events(db, crops)
    assert [p.name for p in crops.iterdir()] == ["user.txt"]


def test_retention_and_missing_image_recovery(tmp_path):
    _, db, crops = data_paths(tmp_path / "private")
    init_db(db)
    save_event(
        db,
        crops,
        "unfamiliar",
        None,
        0.2,
        np.zeros((8, 8, 3), np.uint8),
        captured_at=datetime.now(timezone.utc) - timedelta(days=10),
    )
    for _ in range(3):
        save_event(db, crops, "familiar", "alice", 0.9, None)
    maintain_events(db, crops, retention_days=7, max_events=2)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    assert list(crops.iterdir()) == []
    save_event(db, crops, "unfamiliar", None, 0.1, np.zeros((8, 8, 3), np.uint8))
    next(crops.glob("*.jpg")).unlink()
    maintain_events(db, crops)
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT count(*) FROM events WHERE image_path IS NOT NULL").fetchone()[0]
            == 0
        )


def test_profile_updates_preserve_other_profiles_and_validate_model(tmp_path):
    path = tmp_path / "gallery.npz"
    save_profile(path, "alice", [np.ones(2)], "model")
    save_profile(path, "bob", [np.array([1, 0])], "model")
    assert set(load_gallery(path, "model")) == {"alice", "bob"}
    with pytest.raises(ValueError, match="replacement"):
        save_profile(path, "alice", [np.ones(2)], "model")
    with pytest.raises(ValueError, match="model mismatch"):
        save_profile(path, "charlie", [np.ones(2)], "different")
    assert len(load_gallery(path, "model")) == 2
    with pytest.raises(ValueError):
        save_gallery(path, {"__schema__": [np.ones(2)]})


def test_reject_symlinks(tmp_path):
    if os.name == "nt":
        pytest.skip("symlink privileges vary on Windows")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        data_paths(link)


def monitor_args():
    return SimpleNamespace(
        det_size=640,
        face_model_root=Path("/not-used"),
        person_model=Path("/not-used.onnx"),
        person_confidence=0.1,
        person_sha256=None,
        track_max_age=2,
        threshold=0.45,
        evidence_ttl=2,
        ambiguity_margin=0.05,
        cooldown=30,
        familiar_log_interval=60,
        recognition_interval=0,
        retention_days=7,
        max_events=10000,
        reconnect_attempts=2,
        camera=0,
        width=320,
        height=240,
        headless=True,
        max_frames=4,
        max_frame_age=1,
        metrics=False,
    )


def test_monitor_end_to_end_without_camera_models_or_gui(tmp_path, monkeypatch):
    from test_pipeline import Detector

    gallery, db, crops = data_paths(tmp_path / "private")
    save_gallery(gallery, {"alice": [np.array([1.0, 0.0])]}, model_id="test")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_args, **_kw: Detector())
    monkeypatch.setattr("security_ai.pipeline.quality_ok", lambda *_: True)
    # Supply quality through the pipeline boundary, not image-based ML behavior.
    original = cli.RecognitionPipeline
    monkeypatch.setattr(
        cli, "RecognitionPipeline", lambda *a: original(*a, quality=lambda *_: True)
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    now = cli.time.monotonic()
    packets = iter(CapturedFrame(frame, now + t, 1) for t in (-0.3, -0.2, -0.1, 0))
    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream
    stream.read.side_effect = lambda: next(packets)
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *a, **kw: stream)
    monkeypatch.setattr(cli.cv2, "imshow", Mock(side_effect=AssertionError("GUI forbidden")))
    monkeypatch.setattr(
        cli.cv2, "VideoCapture", Mock(side_effect=AssertionError("Camera forbidden"))
    )
    cli.monitor(monitor_args(), gallery, db, crops)
    stream.close.assert_called_once()
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT status,profile_name,track_id,session_id,image_path FROM events ORDER BY id"
        ).fetchall()
    assert [r[0] for r in rows] == ["familiar", "unfamiliar"]
    assert rows[0][1] == "alice" and rows[1][1] is None
    assert rows[0][2] != rows[1][2] and rows[0][3] == rows[1][3]
    assert rows[0][4] is None and Path(rows[1][4]).exists()
    assert not frame.any()  # No annotation leaked into raw evidence.


def test_storage_failure_does_not_acknowledge_event():
    p = pipeline(minimum=1)
    frame = np.zeros((240, 320, 3), np.uint8)
    assert all(r.emit for r in p.process(frame, 0))
    # A failed writer does not call acknowledge; the next observation may retry.
    assert all(r.emit for r in p.process(frame, 0.1))


def test_legacy_gallery_requires_explicit_reenrollment(tmp_path):
    path = tmp_path / "gallery.npz"
    np.savez(path, alice=np.ones((1, 2)))
    assert "alice" in load_gallery(path)
    with pytest.raises(ValueError, match="legacy"):
        load_gallery(path, "new-model")


def test_local_analyzer_refuses_missing_weights(tmp_path):
    from security_ai.vision import create_analyzer

    with pytest.raises(ValueError, match="no automatic downloads"):
        create_analyzer(root=tmp_path)


def test_monitor_retries_failed_event_write(tmp_path, monkeypatch):
    from test_pipeline import Detector

    gallery, db, crops = data_paths(tmp_path / "private")
    save_gallery(gallery, {"alice": [np.array([1.0, 0.0])]}, model_id="test")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_kw: Detector())
    original = cli.RecognitionPipeline
    monkeypatch.setattr(
        cli, "RecognitionPipeline", lambda *a: original(*a, quality=lambda *_: True)
    )
    frame = np.zeros((240, 320, 3), np.uint8)
    now = cli.time.monotonic()
    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream
    packets = iter(CapturedFrame(frame, now + t, 1) for t in (-0.3, -0.2, -0.1, 0))
    stream.read.side_effect = lambda: next(packets)
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *a, **kw: stream)
    real_save = cli.save_event
    calls = []

    def fail_once(*a, **kw):
        calls.append(a[2])
        if len(calls) == 1:
            raise OSError("disk full")
        return real_save(*a, **kw)

    monkeypatch.setattr(cli, "save_event", fail_once)
    cli.monitor(monitor_args(), gallery, db, crops)
    assert calls == ["familiar", "unfamiliar", "familiar"]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2


def test_reconnect_discards_old_identity_history(tmp_path, monkeypatch):
    from test_pipeline import Detector

    gallery, db, crops = data_paths(tmp_path / "private")
    save_gallery(gallery, {"alice": [np.array([1.0, 0.0])]}, model_id="test")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_kw: Detector())
    original = cli.RecognitionPipeline
    monkeypatch.setattr(
        cli, "RecognitionPipeline", lambda *a: original(*a, quality=lambda *_: True)
    )
    frame = np.zeros((240, 320, 3), np.uint8)
    now = cli.time.monotonic()
    # Two observations before reconnect and two after must never form a decision.
    packets = iter(
        CapturedFrame(frame, now + t, g) for t, g in [(-0.3, 1), (-0.2, 1), (-0.1, 2), (0, 2)]
    )
    stream = Mock(dropped=0, reconnects=1)
    stream.start.return_value = stream
    stream.read.side_effect = lambda: next(packets)
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *a, **kw: stream)
    cli.monitor(monitor_args(), gallery, db, crops)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0


def test_enrollment_cancel_and_success_are_camera_free(tmp_path, monkeypatch):
    gallery, _, _ = data_paths(tmp_path / "private")
    model = SimpleNamespace(
        fingerprint="test", detect=lambda f: [SimpleNamespace(bbox=np.array([0, 0, 8, 8]))]
    )
    embeddings = iter([np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.8, 0.6])])
    model.embed = lambda *_: next(embeddings)
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "quality_ok", lambda *_: True)
    stream = Mock()
    stream.start.return_value = stream
    stream.read.side_effect = lambda: CapturedFrame(
        np.zeros((8, 8, 3), np.uint8), cli.time.monotonic(), 1
    )
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *a, **kw: stream)
    monkeypatch.setattr(cli.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)
    args = monitor_args()
    args.name, args.samples, args.replace = "alice", 2, False
    args.enrollment_consistency, args.duplicate_threshold = 0.6, 0.995
    keys = iter([32, ord("q")])
    monkeypatch.setattr(cli.cv2, "waitKey", lambda *_: next(keys))
    cli.enroll(args, gallery)
    assert not gallery.exists()
    keys = iter([32, 32])
    monkeypatch.setattr(cli.cv2, "waitKey", lambda *_: next(keys))
    cli.enroll(args, gallery)
    assert len(load_gallery(gallery, "test")["alice"]) == 2
    assert stream.close.call_count == 2


def test_commit_failure_removes_final_crop_and_rolls_back(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from security_ai import storage

    _, db, crops = data_paths(tmp_path / "private")
    init_db(db)
    original = storage.database

    @contextmanager
    def failed_commit(path):
        with original(path) as conn:
            yield conn
            raise sqlite3.OperationalError("simulated commit failure")

    monkeypatch.setattr(storage, "database", failed_commit)
    with pytest.raises(sqlite3.OperationalError, match="commit failure"):
        save_event(db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8))
    assert not list(crops.iterdir())
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
