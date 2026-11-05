"""Local health, bounded aggregate metrics and privacy-safe structured logs."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Iterator
import ctypes
import json
import math
import os
import signal
import sys
import tempfile
import time
import numpy as np
from .storage import private_directory, private_file


def log_event(event: str, *, reason: str | None = None) -> None:
    # Callers supply static codes or exception class names, never exception text,
    # face names, embeddings, images, model paths or credentials.
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event}
    if reason is not None:
        record["reason"] = reason
    print(json.dumps(record), file=sys.stderr, flush=True)


def memory_mb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            return (
                int(Path("/proc/self/statm").read_text().split()[1])
                * os.sysconf("SC_PAGE_SIZE")
                / 1024**2
            )
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return counters.working / 1024**2
    return None


class RuntimeMetrics:
    def __init__(
        self, memory_budget_mb: int = 2048, cpu_budget_percent: float = 75.0, window: int = 256
    ) -> None:
        if (
            memory_budget_mb <= 0
            or not math.isfinite(cpu_budget_percent)
            or not 0 < cpu_budget_percent <= 100
            or window < 1
        ):
            raise ValueError("Invalid runtime budgets or metrics window")
        self.counts: Counter = Counter()
        self.samples: dict[str, deque] = {}
        self.window = window
        self.faults: set[str] = set()
        self.memory_budget_mb, self.cpu_budget = memory_budget_mb, cpu_budget_percent
        self._wall, self._cpu = time.monotonic(), time.process_time()

    def observe(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Timing samples must be finite and nonnegative")
            self.samples.setdefault(name, deque(maxlen=self.window)).append(value)

    def snapshot(self, state: str = "healthy") -> dict:
        wall, cpu = time.monotonic(), time.process_time()
        percent = max(
            0.0,
            (cpu - self._cpu) / max(wall - self._wall, 1e-9) * 100 / max(os.cpu_count() or 1, 1),
        )
        self._wall, self._cpu = wall, cpu
        memory = memory_mb()
        for name, exceeded in (
            ("memory_budget", memory is not None and memory > self.memory_budget_mb),
            ("cpu_budget", percent > self.cpu_budget),
        ):
            if exceeded:
                self.faults.add(name)
            else:
                self.faults.discard(name)
        return {
            "schema": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "degraded" if state == "healthy" and self.faults else state,
            "faults": sorted(self.faults),
            "counts": dict(self.counts),
            "memory_mb": memory,
            "normalized_cpu_percent": percent,
            "timings": {
                name: {
                    "count": len(values),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                }
                for name, values in self.samples.items()
            },
        }


class HealthReporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        private_directory(path.parent)

    def write(self, report: dict) -> None:
        private_file(self.path)
        pending = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, prefix=".health-", delete=False
            ) as stream:
                pending = Path(stream.name)
                json.dump(report, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(self.path)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)


def read_health(path: Path, max_age: float = 10.0) -> dict:
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("Maximum health age must be positive")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(data["updated_at"])
        if timestamp.tzinfo is None:
            raise ValueError("Missing timezone")
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        state = data["state"]
        if state not in ("starting", "healthy", "degraded", "stopped", "failed"):
            raise ValueError("Invalid state")
        valid = data["schema"] == 1 and 0 <= age <= max_age
        return {
            "healthy": valid and state == "healthy",
            "state": state,
            "age_seconds": round(age, 3),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {"healthy": False, "state": "unavailable", "age_seconds": None}


@contextmanager
def shutdown_signals() -> Iterator[Event]:
    stop = Event()
    previous = {}

    def request_stop(signum: int, frame: object) -> None:
        stop.set()

    try:
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, request_stop)
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
