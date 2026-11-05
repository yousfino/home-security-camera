"""Small injectable boundaries; importing these never opens a camera or model."""

from __future__ import annotations

from typing import Any, Protocol
import numpy as np


class Capture(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class FaceAnalyzer(Protocol):
    def get(self, frame: np.ndarray) -> list[Any]: ...


class Display(Protocol):
    def show(self, frame: np.ndarray) -> bool:
        """Return False when the user requests shutdown."""
        ...

    def close(self) -> None: ...
