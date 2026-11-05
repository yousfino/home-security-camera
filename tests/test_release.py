"""Synthetic multi-person soak and process-level shutdown; never opens hardware."""

from datetime import datetime, timezone
from types import SimpleNamespace
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import numpy as np
import pytest
from security_ai.config import RecognitionConfig
from security_ai.pipeline import RecognitionPipeline
from security_ai.tracking import PersonTracker
from security_ai.detection import PersonDetection
from security_ai.episodes import EpisodeManager
from security_ai.storage import init_db, save_event, save_episodes
from security_ai.operations import RuntimeMetrics


def test_ten_person_synthetic_soak_bounds_state_and_deduplicates(tmp_path):
    class People:
        def detect(self, frame):
            return [PersonDetection((i * 100, 0, i * 100 + 90, 200), 0.9) for i in range(10)]

    class Faces:
        def detect(self, frame):
            return [
                SimpleNamespace(bbox=np.array([i * 100 + 10, 10, i * 100 + 50, 60]), number=i)
                for i in range(10)
            ]

        def embed(self, frame, face):
            return np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]][face.number % 3])

    pipeline = RecognitionPipeline(
        People(),
        PersonTracker(),
        Faces(),
        {"a": [np.array([1.0, 0.0])], "b": [np.array([-1.0, 0.0])]},
        RecognitionConfig(recognition_interval=0.1),
        quality=lambda f, face: face.number != 9,
    )
    episodes = EpisodeManager()
    db, crops = tmp_path / "events.db", tmp_path / "crops"
    init_db(db)
    wall = datetime.now(timezone.utc)
    metrics = RuntimeMetrics()
    frame = np.zeros((240, 1000, 3), np.uint8)
    for index in range(400):
        timestamp = index * 0.05
        decisions = episodes.update(pipeline.process(frame, timestamp), timestamp)
        save_episodes(db, episodes.pending(), "test", wall, 0)
        episodes.checkpointed()
        for decision in decisions:
            if decision.emit:
                save_event(
                    db,
                    crops,
                    decision.status,
                    decision.name,
                    decision.score,
                    None,
                    track_id=decision.track_id,
                    episode_id=decision.episode_id,
                    event_key=decision.event_key,
                )
                pipeline.acknowledge(decision, timestamp)
                episodes.acknowledge(decision)
        metrics.observe(pipeline.timings)
    assert len(pipeline.states) == 10 and len(episodes.episodes) == 10
    assert len(metrics.samples["total_ms"]) == 256
    assert not pipeline.overflow_people
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 9
        assert (
            conn.execute("SELECT count(*) FROM events WHERE status='unfamiliar'").fetchone()[0] == 3
        )
        assert conn.execute("SELECT count(DISTINCT episode_id) FROM events").fetchone()[0] == 9
    assert not list(crops.glob("*.jpg"))


@pytest.mark.skipif(
    os.name == "nt", reason="Windows terminate force-kills; signal handler is tested separately"
)
def test_sigterm_stops_owned_mock_monitor_process(tmp_path):
    # All inference/capture factories are replaced before CLI composition.
    script = r"""
import sys,time
from types import SimpleNamespace
import numpy as np
from security_ai import cli
class Camera:
    def read(self):
        time.sleep(.005)
        return True,np.zeros((32,32,3),np.uint8)
    def release(self):pass
cli.camera=lambda *_:Camera()
cli.create_analyzer=lambda *_:SimpleNamespace(fingerprint='synthetic',detect=lambda _:[])
cli.YoloPersonDetector=lambda *_a,**_k:SimpleNamespace(detect=lambda _:[])
sys.argv=['security-ai','--data-dir',sys.argv[1],'monitor','--person-model','unused.onnx',
          '--headless','--max-runtime','30']
cli.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path / "private")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        ready = False
        while time.monotonic() < deadline and process.poll() is None:
            try:
                ready = (
                    json.loads((tmp_path / "private" / "health.json").read_text())["state"]
                    == "healthy"
                )
            except (OSError, ValueError, KeyError):
                pass
            if ready:
                break
            time.sleep(0.05)
        assert ready, "Mock process did not reach healthy state"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert json.loads((tmp_path / "private" / "health.json").read_text())["state"] == "stopped"
        assert "monitor_stopped" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
