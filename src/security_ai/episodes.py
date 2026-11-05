"""Conservative person episodes. Track continuity is never biometric proof."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from uuid import uuid4
from .pipeline import TrackDecision


@dataclass
class Episode:
    episode_id: str
    track_id: int
    started: float
    last_seen: float
    box: tuple[float, float, float, float]
    state: str = "active"
    identity: str | None = None
    emitted: set[tuple[str, str | None]] = field(default_factory=set)


def overlap(a: tuple, b: tuple) -> float:
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


class EpisodeManager:
    def __init__(self, gap: float = 2.0, reacquire_iou: float = 0.3) -> None:
        if not math.isfinite(gap) or gap <= 0 or not 0 < reacquire_iou <= 1:
            raise ValueError("Require positive episode gap and reacquisition IoU in (0,1]")
        self.gap, self.reacquire_iou = gap, reacquire_iou
        self.episodes: dict[str, Episode] = {}
        self.by_track: dict[int, str] = {}
        self.dirty: set[str] = set()

    def close(self) -> None:
        for episode in self.episodes.values():
            if episode.state != "closed":
                episode.state = "closed"
                self.dirty.add(episode.episode_id)
        self.by_track.clear()

    def pending(self) -> list[Episode]:
        return [self.episodes[key] for key in self.dirty]

    def checkpointed(self) -> None:
        self.dirty.clear()
        for key in [key for key, e in self.episodes.items() if e.state == "closed"]:
            self.episodes.pop(key)
        self.by_track = {t: key for t, key in self.by_track.items() if key in self.episodes}

    def acknowledge(self, decision: TrackDecision) -> None:
        if decision.emit and decision.episode_id in self.episodes:
            self.episodes[decision.episode_id].emitted.add((decision.status, decision.name))

    def update(self, decisions: list[TrackDecision], timestamp: float) -> list[TrackDecision]:
        visible = {d.track_id for d in decisions}
        for episode in self.episodes.values():
            if episode.state == "closed":
                continue
            state = (
                "closed"
                if timestamp - episode.last_seen > self.gap
                else "lost"
                if episode.track_id not in visible
                else "active"
            )
            if state != episode.state:
                episode.state = state
                self.dirty.add(episode.episode_id)
        result = []
        for decision in decisions:
            key = self.by_track.get(decision.track_id)
            episode = self.episodes.get(key)
            if episode is None or episode.state == "closed":
                episode = Episode(
                    uuid4().hex, decision.track_id, timestamp, timestamp, decision.box
                )
                self.episodes[episode.episode_id] = episode
                self.by_track[decision.track_id] = episode.episode_id
                self.dirty.add(episode.episode_id)
            # A new provisional episode may join exactly one lost episode only
            # after fresh stable familiar evidence, never from appearance alone.
            if decision.status == "familiar" and not episode.emitted:
                candidates = [
                    e
                    for e in self.episodes.values()
                    if e.state == "lost"
                    and e.track_id not in visible
                    and e.identity == decision.name
                    and timestamp - e.last_seen <= self.gap
                    and overlap(e.box, decision.box) >= self.reacquire_iou
                ]
                # Two visible instances of the same identity make reattachment ambiguous.
                same_name = sum(
                    d.status == "familiar" and d.name == decision.name for d in decisions
                )
                if len(candidates) == 1 and same_name == 1:
                    episode.state = "closed"
                    self.dirty.add(episode.episode_id)
                    episode = candidates[0]
                    self.by_track.pop(episode.track_id, None)
                    episode.track_id = decision.track_id
                    self.by_track[decision.track_id] = episode.episode_id
            if episode.state != "active":
                self.dirty.add(episode.episode_id)
            episode.state = "active"
            episode.last_seen = timestamp
            episode.box = decision.box
            if decision.status == "familiar":
                episode.identity = decision.name
            elif decision.status == "unfamiliar":
                episode.identity = None
            emit = decision.emit and (decision.status, decision.name) not in episode.emitted
            event_key = (
                f"{episode.episode_id}:{decision.status}:{decision.name or '-'}" if emit else None
            )
            result.append(
                replace(decision, episode_id=episode.episode_id, event_key=event_key, emit=emit)
            )
        return result
