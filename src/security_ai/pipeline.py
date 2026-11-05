"""Headless per-person recognition, independent of capture, GUI and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import time
import numpy as np

from .config import RecognitionConfig
from .core import DecisionWindow, GalleryIndex, Observation, classify
from .detection import PersonDetector
from .interfaces import FaceAnalyzer
from .tracking import PersonTracker, associate_faces
from .vision import quality_ok


@dataclass(frozen=True)
class TrackDecision:
    track_id: int
    box: tuple[float, float, float, float]
    status: str = "unresolved"
    name: str | None = None
    score: float = -1.0
    face: Any = None
    emit: bool = False
    episode_id: str | None = None
    event_key: str | None = None


@dataclass
class TrackState:
    window: DecisionWindow
    last_seen: float
    last_evidence: float | None = None
    logged: dict[tuple[str, str | None], float] = field(default_factory=dict)


class RecognitionPipeline:
    def __init__(
        self,
        detector: PersonDetector,
        tracker: PersonTracker,
        analyzer: FaceAnalyzer,
        gallery: dict[str, list[np.ndarray]],
        config: RecognitionConfig,
        quality: Callable[[np.ndarray, Any], bool] = quality_ok,
    ) -> None:
        self.detector, self.tracker, self.analyzer = detector, tracker, analyzer
        self.gallery, self.config, self.quality = (
            GalleryIndex(gallery, config.aggregation),
            config,
            quality,
        )
        self.states: dict[int, TrackState] = {}
        self.timings: dict[str, float] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.states.clear()

    def acknowledge(self, decision: TrackDecision, timestamp: float) -> None:
        """Advance cooldown only after persistence succeeds."""
        state = self.states.get(decision.track_id)
        if state is not None and decision.emit:
            state.logged[(decision.status, decision.name)] = timestamp

    def process(self, frame: np.ndarray, timestamp: float) -> list[TrackDecision]:
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        detected = time.perf_counter()
        tracks = self.tracker.update(detections, timestamp)
        tracked = time.perf_counter()
        self.states = {
            key: state
            for key, state in self.states.items()
            if timestamp - state.last_seen <= self.tracker.max_age
        }
        faces = self.analyzer.detect(frame) if tracks else []
        faces_detected = time.perf_counter()
        associations = associate_faces(tracks, faces)
        by_track = {track_id: faces[i] for i, track_id in associations.items()}
        results = []
        for track in tracks:
            state = self.states.setdefault(
                track.track_id,
                TrackState(DecisionWindow(self.config.minimum, self.config.window), timestamp),
            )
            state.last_seen = timestamp
            face = by_track.get(track.track_id)
            if (
                state.last_evidence is not None
                and timestamp - state.last_evidence > self.config.evidence_ttl
            ):
                state.window.clear()
            status, name, score = "unresolved", None, -1.0
            if self.gallery.names and face is not None and self.quality(frame, face):
                if (
                    state.last_evidence is not None
                    and timestamp - state.last_evidence < self.config.recognition_interval
                ):
                    # No new identity observation or event on skipped inference.
                    results.append(TrackDecision(track.track_id, track.box, face=face))
                    continue
                embedding = self.analyzer.embed(frame, face)
                name, score, runner_up = self.gallery.match(embedding)
                status = classify(name, score, self.config.threshold, True)
                if status == "familiar" and score - runner_up < self.config.ambiguity_margin:
                    status = "unresolved"
                state.last_evidence = timestamp
            label, stable_name = state.window.add(
                Observation(status, name if status == "familiar" else None, score)
            )
            interval = (
                self.config.cooldown if label == "unfamiliar" else self.config.familiar_log_interval
            )
            last = state.logged.get((label, stable_name), float("-inf"))
            emit = label != "unresolved" and timestamp - last >= interval
            results.append(
                TrackDecision(track.track_id, track.box, label, stable_name, score, face, emit)
            )
        finished = time.perf_counter()
        self.timings = {
            "person_ms": (detected - started) * 1000,
            "tracking_ms": (tracked - detected) * 1000,
            "face_detection_ms": (faces_detected - tracked) * 1000,
            "recognition_ms": (finished - faces_detected) * 1000,
            "total_ms": (finished - started) * 1000,
        }
        return results
