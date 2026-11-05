"""Local score evaluation. Reports aggregates only; never changes runtime policy."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable


def load_scores(path: Path) -> list[tuple[float, bool]]:
    samples = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["score", "same_identity"]:
            raise ValueError("CSV header must be exactly score,same_identity; no identity names")
        for line, row in enumerate(reader, 2):
            if set(row) != {"score", "same_identity"}:
                raise ValueError(f"Unexpected columns at row {line}")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid score at row {line}") from exc
            if (
                not math.isfinite(score)
                or not -1 <= score <= 1
                or row["same_identity"] not in ("0", "1")
            ):
                raise ValueError(f"Invalid score/label at row {line}")
            samples.append((score, row["same_identity"] == "1"))
    if not samples or not any(label for _, label in samples) or all(label for _, label in samples):
        raise ValueError("Require both genuine and impostor comparisons")
    return samples


def wilson(errors: int, count: int) -> tuple[float, float]:
    if count < 1 or not 0 <= errors <= count:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    rate = errors / count
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = (
        z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate(
    samples: Iterable[tuple[float, bool]], thresholds: list[float], max_fmr: float = 0.01
) -> dict:
    rows = list(samples)
    genuine = [score for score, label in rows if label]
    impostor = [score for score, label in rows if not label]
    if not genuine or not impostor:
        raise ValueError("Require genuine and impostor comparisons")
    if not math.isfinite(max_fmr) or not 0 < max_fmr < 1:
        raise ValueError("Maximum false-match rate must be in (0,1)")
    if not thresholds or any(not math.isfinite(t) or not -1 <= t <= 1 for t in thresholds):
        raise ValueError("Thresholds must be finite cosine scores in [-1,1]")
    if any(not math.isfinite(score) or not -1 <= score <= 1 for score, _ in rows):
        raise ValueError("Scores must be finite and in [-1,1]")
    results = []
    for threshold in sorted(set(thresholds)):
        fm = sum(s >= threshold for s in impostor)
        fnm = sum(s < threshold for s in genuine)
        results.append(
            {
                "threshold": threshold,
                "false_matches": fm,
                "false_non_matches": fnm,
                "fmr": fm / len(impostor),
                "fnmr": fnm / len(genuine),
                "fmr_wilson95": wilson(fm, len(impostor)),
                "fnmr_wilson95": wilson(fnm, len(genuine)),
            }
        )
    eligible = [r for r in results if r["fmr_wilson95"][1] <= max_fmr]
    selected = min(eligible, key=lambda r: (r["fnmr"], -r["threshold"])) if eligible else None
    return {
        "genuine_count": len(genuine),
        "impostor_count": len(impostor),
        "results": results,
        "suggested_threshold": None if selected is None else selected["threshold"],
        "target_fmr": max_fmr,
        "runtime_configuration_changed": False,
        "limitations": [
            "Enrollment-independent samples and held-out evaluation are required.",
            "Wilson intervals assume independent trials; repeated frames violate this.",
            "Pairwise cosine evaluation does not validate full open-set identification, "
            "quality rejection, ambiguity margins or temporal/event decisions.",
        ],
    }
