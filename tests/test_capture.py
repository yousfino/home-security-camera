from threading import Event
import numpy as np
import pytest
from security_ai.capture import LatestFrameCapture
from test_pipeline import pipeline


class FakeCamera:
    def __init__(self):
        self.closed = False
        self.calls = 0

    def read(self):
        self.calls += 1
        return True, np.full((8, 8, 3), self.calls % 256, dtype=np.uint8)

    def release(self):
        self.closed = True


def test_latest_only_buffer_and_shutdown():
    cap = FakeCamera()
    stream = LatestFrameCapture(lambda: cap).start()
    try:
        packet = stream.read(timeout=1)
        assert packet is not None and packet.generation == 1
        assert stream.dropped > 0 or cap.calls >= 1
    finally:
        stream.close()
    assert cap.closed


def test_reconnect_generation_and_retry_limit():
    bad = FakeCamera()
    bad.read = lambda: (False, None)
    good = FakeCamera()
    cameras = iter([bad, good])
    stream = LatestFrameCapture(lambda: next(cameras), initial_backoff=0.001).start()
    try:
        packet = stream.read(timeout=1)
        assert packet.generation == 2 and bad.closed
    finally:
        stream.close()
    stream = LatestFrameCapture(
        lambda: (_ for _ in ()).throw(RuntimeError("unplugged")),
        max_attempts=2,
        initial_backoff=0.001,
    ).start()
    try:
        with pytest.raises(RuntimeError, match="2 consecutive"):
            stream.read(timeout=1)
    finally:
        stream.close()


def test_shutdown_interrupts_backoff():
    attempted = Event()

    def fail():
        attempted.set()
        raise RuntimeError("unplugged")

    stream = LatestFrameCapture(fail, initial_backoff=10).start()
    assert attempted.wait(1)
    stream.close(timeout=0.5)


def test_inference_interval_never_replays_event():
    p = pipeline(recognition_interval=0.5, minimum=1)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    results = p.process(frame, 0)
    assert all(r.emit for r in results)
    results = p.process(frame, 0.1)
    assert all(not r.emit and r.status == "unresolved" for r in results)
    assert all(r.emit for r in p.process(frame, 0.6))
    assert p.timings["total_ms"] >= p.timings["person_ms"]
