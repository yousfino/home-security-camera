"""ByteTrack adapter with wall-clock expiry and session-local public IDs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from .detection import PersonDetection


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float


class PersonTracker:
    def __init__(self, max_age: float = 2.0, frame_rate: int = 15) -> None:
        if not math.isfinite(max_age) or max_age <= 0 or frame_rate <= 0:
            raise ValueError("Require positive track age and frame rate")
        # ByteTrack is optional for Sprint 1 users; import only when requested.
        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate, lost_track_buffer=max(1, round(max_age * 30))
        )
        self.max_age = max_age
        self._last: float | None = None
        self._seen: dict[int, float] = {}
        self._ids: dict[int, int] = {}
        self._next = 1

    def reset(self) -> None:
        self._tracker.reset()
        self._seen.clear()
        self._ids.clear()
        self._last = None
        # Public IDs never reuse an earlier ID during this process.

    def update(self, detections: list[PersonDetection], timestamp: float) -> list[PersonTrack]:
        if not math.isfinite(timestamp) or (self._last is not None and timestamp <= self._last):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        if self._last is not None and timestamp - self._last > self.max_age:
            self.reset()
        expired = {key for key, seen in self._seen.items() if timestamp - seen > self.max_age}
        # Drop expired internal tracks before matching, not just their public labels.
        for attr in ("tracked_tracks", "lost_tracks"):
            setattr(
                self._tracker,
                attr,
                [t for t in getattr(self._tracker, attr) if t.external_track_id not in expired],
            )
        for key in expired:
            self._seen.pop(key, None)
            self._ids.pop(key, None)
        # The backend retains removed tracks for bookkeeping; no application needs them.
        self._tracker.removed_tracks.clear()
        self._last = timestamp
        data = self._sv.Detections(
            xyxy=np.asarray([d.box for d in detections], dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray([d.confidence for d in detections], dtype=np.float32),
            class_id=np.zeros(len(detections), dtype=int),
        )
        tracked = self._tracker.update_with_detections(data)
        result = []
        for box, confidence, internal_id in zip(
            tracked.xyxy, tracked.confidence, tracked.tracker_id
        ):
            key = int(internal_id)
            if key not in self._ids:
                self._ids[key] = self._next
                self._next += 1
            self._seen[key] = timestamp
            result.append(
                PersonTrack(self._ids[key], tuple(float(v) for v in box), float(confidence))
            )
        return result


def associate_faces(tracks: list[PersonTrack], faces: list[object]) -> dict[int, int]:
    """Map face index to track ID only for an unambiguous one-to-one containment.

    Overlapping persons and multiple faces inside one person remain unresolved.
    Tracking is geometric continuity, never proof of biometric identity.
    """
    candidates: dict[int, int] = {}
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = map(float, face.bbox)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or not np.isfinite([x1, y1, x2, y2]).all():
            continue
        matches = []
        for track in tracks:
            a, b, c, d = track.box
            intersection = max(0.0, min(x2, c) - max(x1, a)) * max(0.0, min(y2, d) - max(y1, b))
            if intersection / area >= 0.9:
                matches.append(track.track_id)
        if len(matches) == 1:
            candidates[index] = matches[0]
    values = list(candidates.values())
    return {i: track_id for i, track_id in candidates.items() if values.count(track_id) == 1}
