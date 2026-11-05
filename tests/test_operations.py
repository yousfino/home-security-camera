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


def test_native_memory_measurement_is_available_on_supported_hosts():
    import sys
    from security_ai.operations import memory_mb

    if sys.platform.startswith("linux") or sys.platform == "win32":
        assert memory_mb() is not None and memory_mb() > 0


def test_maximum_runtime_ends_camera_outage_without_hardware(tmp_path, monkeypatch):
    gallery, db, crops = data_paths(tmp_path / "private")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_k: Detector())
    clock = [0.0]

    def now():
        clock[0] += 0.01
        return clock[0]

    monkeypatch.setattr(cli.time, "monotonic", now)
    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream
    stream.read.return_value = None
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *_a, **_k: stream)
    args = monitor_args()
    args.max_runtime = 0.2
    args.max_frames = None
    cli.monitor(args, gallery, db, crops)
    assert 0 < stream.read.call_count < 20
    stream.close.assert_called_once()


def test_startup_failure_is_visible_in_health_without_exception_text(tmp_path, monkeypatch):
    gallery, db, crops = data_paths(tmp_path / "private")

    def fail(*_):
        raise ValueError("private name and private path must not enter health")

    monkeypatch.setattr(cli, "create_analyzer", fail)
    with pytest.raises(ValueError):
        cli.monitor(monitor_args(), gallery, db, crops)
    report = json.loads((db.parent / "health.json").read_text())
    assert report["state"] == "failed"
    assert "private name" not in json.dumps(report)


def test_stale_frames_do_not_disrupt_periodic_retention(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from security_ai.capture import CapturedFrame

    gallery, db, crops = data_paths(tmp_path / "private")
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: SimpleNamespace(fingerprint="test"))
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_k: object())
    fake_pipeline = SimpleNamespace(
        reset=lambda: None, process=lambda *_: [], timings={}, overflow_people=0
    )
    monkeypatch.setattr(cli, "RecognitionPipeline", lambda *_: fake_pipeline)
    maintenance = Mock()
    monkeypatch.setattr(cli, "maintain_events", maintenance)
    now = cli.time.monotonic()
    count = 0

    def next_frame():
        nonlocal count
        count += 1
        return CapturedFrame(np.zeros((8, 8, 3), np.uint8), 0 if count == 1 else now, 1)

    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream
    stream.read.side_effect = next_frame
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *_a, **_k: stream)
    args = monitor_args()
    args.max_frames = 301
    args.max_frame_age = 10
    cli.monitor(args, gallery, db, crops)
    assert maintenance.call_count == 2  # startup plus 300 successfully processed frames
