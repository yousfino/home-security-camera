"""Pure recognition and decision helpers, suitable for unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


def normalized(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Invalid or zero embedding")
    return v / norm


def best_match(
    embedding: np.ndarray, gallery: dict[str, list[np.ndarray]]
) -> tuple[str | None, float]:
    query = normalized(embedding)
    best_name, best_score = None, -1.0
    for name, samples in gallery.items():
        for sample in samples:
            reference = normalized(sample)
            if reference.shape != query.shape:
                continue
            score = float(np.dot(query, reference))
            if score > best_score:
                best_name, best_score = name, score
    return best_name, best_score


def classify(name: str | None, score: float, threshold: float, quality_ok: bool) -> str:
    if not quality_ok:
        return "unresolved"
    return "familiar" if name is not None and score >= threshold else "unfamiliar"


@dataclass
class Observation:
    status: str
    name: str | None
    score: float


class DecisionWindow:
    """Require repeated consistent observations for a single-face MVP."""

    def __init__(self, minimum: int = 3, window: int = 5):
        if not 1 <= minimum <= window:
            raise ValueError("Require 1 <= minimum <= window")
        self.minimum = minimum
        self.observations: deque[Observation] = deque(maxlen=window)

    def add(self, obs: Observation) -> tuple[str, str | None]:
        self.observations.append(obs)
        # A rejected frame breaks evidence continuity. Never emit a historical
        # decision against a low-quality current observation.
        if obs.status == "unresolved":
            self.clear()
            return "unresolved", None
        usable = list(self.observations)
        if len(usable) < self.minimum:
            return "unresolved", None
        # Avoid accepting an identity when recent evidence disagrees.
        recent = usable[-self.minimum :]
        if all(o.status == "familiar" and o.name == recent[0].name for o in recent):
            return "familiar", recent[0].name
        if all(o.status == "unfamiliar" for o in recent):
            return "unfamiliar", None
        return "unresolved", None

    def clear(self) -> None:
        self.observations.clear()
