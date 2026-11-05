from dataclasses import replace
import sqlite3
import numpy as np
from security_ai.episodes import EpisodeManager
from security_ai.pipeline import TrackDecision
from security_ai.storage import init_db, save_event


def decision(track=1, status="familiar", name="alice", box=(0, 0, 100, 200)):
    return TrackDecision(track, box, status, name, 0.9, emit=True)


def test_episode_dedup_and_familiar_reacquisition_requires_fresh_evidence():
    manager = EpisodeManager(gap=2)
    first = manager.update([decision()], 0)[0]
    manager.acknowledge(first)
    assert not manager.update([decision()], 0.1)[0].emit
    manager.update([], 0.2)
    provisional = manager.update([decision(2, "unresolved", None)], 0.3)[0]
    assert provisional.episode_id != first.episode_id
    reacquired = manager.update([decision(2)], 0.4)[0]
    assert reacquired.episode_id == first.episode_id and not reacquired.emit
    manager.update([], 3)
    manager.checkpointed()
    reentry = manager.update([decision(3)], 3.1)[0]
    assert reentry.episode_id != first.episode_id and reentry.emit


def test_unknown_people_and_ambiguous_familiar_people_never_join():
    manager = EpisodeManager()
    first = manager.update([decision(1, "unfamiliar", None)], 0)[0]
    manager.acknowledge(first)
    manager.update([], 0.1)
    second = manager.update([decision(2, "unfamiliar", None)], 0.2)[0]
    assert first.episode_id != second.episode_id and second.emit
    manager = EpisodeManager()
    original = manager.update([decision()], 0)[0]
    manager.acknowledge(original)
    manager.update([], 0.1)
    results = manager.update([decision(2), decision(3)], 0.2)
    assert all(r.episode_id != original.episode_id for r in results)


def test_checkpoint_does_not_lose_pending_close_and_reconnect_starts_new_episode():
    manager = EpisodeManager()
    first = manager.update([decision()], 0)[0]
    manager.checkpointed()
    manager.close()
    assert manager.pending()[0].state == "closed"
    manager.checkpointed()
    assert not manager.episodes
    assert manager.update([decision(2)], 1)[0].episode_id != first.episode_id


def test_event_key_is_idempotent_even_if_acknowledgment_was_lost(tmp_path):
    db, crops = tmp_path / "events.db", tmp_path / "crops"
    init_db(db)
    kwargs = {"episode_id": "episode", "event_key": "key"}
    first = save_event(db, crops, "unfamiliar", None, 0.2, np.zeros((8, 8, 3), np.uint8), **kwargs)
    assert save_event(db, crops, "unfamiliar", None, 0.2, None, **kwargs) == first
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert len(list(crops.glob("*.jpg"))) == 1


def test_changed_classification_gets_own_event_without_reusing_identity():
    manager = EpisodeManager()
    first = manager.update([decision()], 0)[0]
    manager.acknowledge(first)
    changed = manager.update([replace(decision(), status="unfamiliar", name=None)], 0.1)[0]
    assert changed.emit and changed.episode_id == first.episode_id
    assert manager.episodes[first.episode_id].identity is None
