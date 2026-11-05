"""Bounded latest-frame capture with reconnects and interruptible backoff.

A worker owns the camera. Consumers never accumulate a frame queue. Native camera
reads may block inside a driver; close reports a timeout instead of waiting forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Event, Thread
from typing import Callable
import math
import time
import numpy as np
from .interfaces import Capture


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    acquired_at: float
    generation: int


class LatestFrameCapture:
    def __init__(
        self,
        factory: Callable[[], Capture],
        max_attempts: int = 8,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1 or not all(
            math.isfinite(v) and v > 0 for v in (initial_backoff, max_backoff)
        ):
            raise ValueError("Require positive attempts and backoff")
        self.factory, self.max_attempts = factory, max_attempts
        self.initial_backoff, self.max_backoff, self.clock = initial_backoff, max_backoff, clock
        self._condition, self._stop = Condition(), Event()
        self._packet: CapturedFrame | None = None
        self._error: Exception | None = None
        self._thread: Thread | None = None
        self.dropped = 0
        self.reconnects = 0

    def start(self) -> LatestFrameCapture:
        if self._thread is not None:
            raise RuntimeError("Capture already started")
        self._thread = Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        cap: Capture | None = None
        attempts, generation = 0, 0
        try:
            while not self._stop.is_set():
                try:
                    if cap is None:
                        cap = self.factory()
                        generation += 1
                        self.reconnects = generation - 1
                    ok, image = cap.read()
                    if not ok or image is None or not image.size:
                        raise RuntimeError("Camera frame read failed")
                    attempts = 0
                    packet = CapturedFrame(image.copy(), self.clock(), generation)
                    with self._condition:
                        if self._packet is not None:
                            self.dropped += 1
                        self._packet = packet
                        self._condition.notify_all()
                except Exception as exc:
                    if cap is not None:
                        cap.release()
                        cap = None
                    # Never expose a buffered pre-outage frame to a consumer.
                    with self._condition:
                        self._packet = None
                    attempts += 1
                    if attempts >= self.max_attempts:
                        raise RuntimeError(
                            f"Camera failed after {attempts} consecutive attempts"
                        ) from exc
                    delay = min(self.max_backoff, self.initial_backoff * 2 ** (attempts - 1))
                    self._stop.wait(delay)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def read(self, timeout: float = 0.1) -> CapturedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._packet is not None or self._error is not None or self._stop.is_set(),
                timeout,
            )
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            packet, self._packet = self._packet, None
            return packet

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("Camera driver did not stop; native read remains blocked")
