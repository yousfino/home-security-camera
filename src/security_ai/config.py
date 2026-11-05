"""Validated configuration shared by the CLI and processing services."""

from __future__ import annotations

import math
from dataclasses import dataclass


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError("Must be a positive integer")
    return result


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Must be finite")
    return result


@dataclass(frozen=True)
class RecognitionConfig:
    threshold: float = 0.45
    aggregation: str = "max"
    minimum: int = 3
    window: int = 5
    evidence_ttl: float = 2.0
    ambiguity_margin: float = 0.05
    recognition_interval: float = 0.2
    cooldown: float = 30.0
    familiar_log_interval: float = 60.0

    def __post_init__(self) -> None:
        if self.aggregation not in ("max", "mean"):
            raise ValueError("Aggregation must be max or mean")
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Cosine threshold must be finite and in [-1, 1]")
        if not 1 <= self.minimum <= self.window:
            raise ValueError("Require 1 <= minimum <= window")
        for name in (
            "evidence_ttl",
            "ambiguity_margin",
            "recognition_interval",
            "cooldown",
            "familiar_log_interval",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.evidence_ttl == 0 or self.ambiguity_margin > 2:
            raise ValueError("Require positive evidence TTL and ambiguity margin <= 2")
