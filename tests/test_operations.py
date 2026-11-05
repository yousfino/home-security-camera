from datetime import datetime, timedelta, timezone
from threading import Event
import json
import signal
from unittest.mock import Mock
import numpy as np
import pytest
from security_ai.operations import (
    RuntimeMetrics,
    HealthReporter,
    read_health,
    log_event,
    shutdown_signals,
)
from security_ai.episodes import EpisodeManager
from security_ai.pipeline import TrackDecision
from security_ai import cli
from security_ai.storage import data_paths, save_gallery, file_lock
from test_pipeline import Analyzer, Detector, pipeline
from test_security_integration import monitor_args


def test_metrics_are_bounded_and_contain_no_biometric_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("security_ai.operations.memory_mb", lambda: 3000.0)
    metrics = RuntimeMetrics(window=3)
    for value in range(10):
        metrics.observe({"frame_age_ms": float(value)})
    report = metrics.snapshot()
    assert report["state"] == "degraded" and "memory_budget" in report["faults"]
    assert report["timings"]["frame_age_ms"]["count"] == 3
    assert report["timings"]["frame_age_ms"]["p50"] == 8
    assert report["timings"]["frame_age_ms"]["p95"] == pytest.approx(8.9)
    assert "name" not in json.dumps(report) and "embedding" not in json.dumps(report)
    path = tmp_path / "private" / "health.json"
    HealthReporter(path).write(report)
    assert not read_health(path)["healthy"]
    report["state"] = "healthy"
    HealthReporter(path).write(report)
    assert read_health(path)["healthy"]
    report["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    HealthReporter(path).write(report)
    assert not read_health(path)["healthy"]
    path.write_text("not json")
    assert read_health(path)["state"] == "unavailable"


def test_logs_only_emit_explicit_static_fields(capsys):
    log_event("event_persistence_failed", reason="OSError")
    record = json.loads(capsys.readouterr().err)
    assert set(record) == {"time", "event", "reason"}
    assert record["reason"] == "OSError"


def test_signal_handler_requests_shutdown_and_restores_handlers():
    previous = signal.getsignal(signal.SIGINT)
    with shutdown_signals() as stop:
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGINT) == previous


def test_monitor_lease_prevents_second_writer(tmp_path):
    gallery, db, crops = data_paths(tmp_path / "private")
    with file_lock(db.with_name("monitor"), blocking=False):
        with pytest.raises((OSError, RuntimeError)):
            cli.monitor(monitor_args(), gallery, db, crops)


def test_shutdown_during_camera_wait_and_final_health(tmp_path, monkeypatch):
    gallery, db, crops = data_paths(tmp_path / "private")
    save_gallery(gallery, {"alice": [np.array([1.0, 0.0])]}, "test")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_k: Detector())
    stop = Event()
    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream

    def wait():
        stop.set()
        return None

    stream.read.side_effect = wait
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *_a, **_k: stream)
    args = monitor_args()
    args.max_frames = None
    cli.monitor(args, gallery, db, crops, stop)
    stream.close.assert_called_once()
    assert read_health(db.parent / "health.json")["state"] == "stopped"


def test_person_capacity_is_explicit_and_bounded():
    p = pipeline(minimum=1)

    class ManyPeople:
        def detect(self, frame):
            from security_ai.detection import PersonDetection

            return [PersonDetection((i * 200, 0, i * 200 + 100, 200), 0.9) for i in range(12)]

    p.detector = ManyPeople()
    results = p.process(np.zeros((240, 2400, 3), np.uint8), 0)
    assert len(results) == 10 and p.overflow_people == 2


def test_checkpoint_failure_has_bounded_episode_memory():
    manager = EpisodeManager(max_episodes=2)

    def decision(track):
        return TrackDecision(track, (0, 0, 100, 200))

    manager.update([decision(1)], 0)
    manager.update([decision(2)], 3)
    with pytest.raises(RuntimeError, match="capacity"):
        manager.update([decision(3)], 6)


def test_embedding_budget_is_fair_across_ten_people():
    from types import SimpleNamespace
    from security_ai.detection import PersonDetection
    from security_ai.pipeline import RecognitionPipeline
    from security_ai.tracking import PersonTracker
    from security_ai.config import RecognitionConfig

    class People:
        def detect(self, frame):
            return [PersonDetection((i * 100, 0, i * 100 + 90, 200), 0.9) for i in range(10)]

    calls = []

    class Faces:
        def detect(self, frame):
            return [
                SimpleNamespace(bbox=np.array([i * 100 + 10, 10, i * 100 + 50, 60]), number=i)
                for i in range(10)
            ]

        def embed(self, frame, face):
            calls.append(face.number)
            return np.array([1.0, 0.0])

    p = RecognitionPipeline(
        People(),
        PersonTracker(),
        Faces(),
        {"a": [np.array([1.0, 0.0])]},
        RecognitionConfig(minimum=1, recognition_interval=0, embeddings_per_frame=2),
        quality=lambda *_: True,
    )
    frame = np.zeros((240, 1000, 3), np.uint8)
    for index in range(5):
        before = len(calls)
        p.process(frame, index * 0.1)
        assert len(calls) - before == 2
    assert set(calls) == set(range(10))
