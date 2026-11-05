"""Private local persistence with schema migration and recoverable image writes.

Data is not encrypted. Use device encryption and a trusted local account.
"""

from __future__ import annotations

from contextlib import contextmanager, closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
import math
import os
import re
import sqlite3
import tempfile
from uuid import uuid4
import cv2
import numpy as np
from .core import GalleryIndex


RESERVED = {"__model__", "__schema__"}


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Data directories must not be symbolic links")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def private_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Data files must not be symbolic links")
    if path.exists():
        path.chmod(0o600)


def data_paths(base: Path) -> tuple[Path, Path, Path]:
    private_directory(base)
    return base / "gallery.npz", base / "events.db", base / "crops"


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Cooperating writers only; local filesystem locking, including Windows."""
    lock = path.with_name(path.name + ".lock")
    private_file(lock)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_gallery(path: Path, expected_model: str | None = None) -> dict[str, list[np.ndarray]]:
    private_file(path)
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        model = str(archive["__model__"]) if "__model__" in archive else "legacy"
        schema = int(archive["__schema__"]) if "__schema__" in archive else 1
        if schema not in (1, 2):
            raise ValueError("Unsupported gallery schema")
        if expected_model is not None and model != expected_model:
            raise ValueError("Gallery model mismatch or legacy gallery; re-enroll profiles")
        gallery = {}
        for key in archive.files:
            if key in RESERVED:
                continue
            safe_profile_name(key)
            values = archive[key]
            if values.ndim != 2 or not all(values.shape):
                raise ValueError("Gallery entries must be nonempty sample matrices")
            gallery[key] = [np.asarray(row, dtype=np.float32) for row in values]
        GalleryIndex(gallery)
        return gallery


def _save_gallery(path: Path, gallery: dict[str, list[np.ndarray]], model_id: str) -> None:
    private_file(path)
    GalleryIndex(gallery)
    arrays = {
        safe_profile_name(name): np.stack(samples).astype(np.float32)
        for name, samples in gallery.items()
    }
    arrays["__model__"] = np.asarray(model_id)
    arrays["__schema__"] = np.asarray(2)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".gallery-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_gallery(
    path: Path, gallery: dict[str, list[np.ndarray]], model_id: str = "legacy"
) -> None:
    with file_lock(path):
        _save_gallery(path, gallery, model_id)


def save_profile(
    path: Path, name: str, samples: list[np.ndarray], model_id: str, replace: bool = False
) -> None:
    with file_lock(path):
        gallery = load_gallery(path, model_id)
        if name in gallery and not replace:
            raise ValueError("Profile already exists; explicit replacement is required")
        gallery[safe_profile_name(name)] = samples
        _save_gallery(path, gallery, model_id)


def safe_profile_name(name: str) -> str:
    if name in RESERVED or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError(
            "Profile names must use 1–40 ASCII letters, numbers, _ or -; reserved names forbidden"
        )
    return name


@contextmanager
def database(path: Path) -> Iterator[sqlite3.Connection]:
    private_file(path)
    # Create a private file before sqlite opens it, regardless of process umask.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.close(fd)
    with closing(sqlite3.connect(path, timeout=2)) as db:
        with db:
            yield db


def init_db(path: Path) -> None:
    with file_lock(path), database(path) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > 3:
            raise ValueError("Unsupported events schema")
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            status TEXT NOT NULL,
            profile_name TEXT,
            similarity REAL,
            image_path TEXT
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
        for name, kind in [
            ("track_id", "INTEGER"),
            ("session_id", "TEXT"),
            ("model_id", "TEXT"),
            ("episode_id", "TEXT"),
            ("event_key", "TEXT"),
        ]:
            if name not in columns:
                db.execute(f"ALTER TABLE events ADD COLUMN {name} {kind}")
        db.execute("CREATE INDEX IF NOT EXISTS events_captured_at ON events(captured_at)")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS event_key_unique ON events(event_key) WHERE event_key IS NOT NULL"
        )
        db.execute("""CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            started_at TEXT NOT NULL, last_seen TEXT NOT NULL, state TEXT NOT NULL)""")
        db.execute("PRAGMA user_version=3")


def save_event(
    db_path: Path,
    crop_dir: Path,
    status: str,
    name: str | None,
    score: float,
    crop: np.ndarray | None,
    *,
    track_id: int | None = None,
    session_id: str | None = None,
    model_id: str | None = None,
    captured_at: datetime | None = None,
    episode_id: str | None = None,
    event_key: str | None = None,
) -> int:
    if (
        status not in ("familiar", "unfamiliar", "unresolved")
        or not math.isfinite(score)
        or not -1 <= score <= 1
    ):
        raise ValueError("Invalid event classification or score")
    if (status == "familiar") != (name is not None):
        raise ValueError("Only familiar events must have a profile name")
    if name is not None:
        safe_profile_name(name)
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise ValueError("Event timestamps must include a timezone")
    private_directory(crop_dir)
    staged: Path | None = None
    final: Path | None = None
    with file_lock(db_path):
        if event_key is not None:
            with database(db_path) as db:
                row = db.execute(
                    "SELECT id,status,profile_name,episode_id FROM events WHERE event_key=?",
                    (event_key,),
                ).fetchone()
                if row is not None:
                    if row[1:] != (status, name, episode_id):
                        raise ValueError("Event key was reused with different event data")
                    return int(row[0])
        try:
            if crop is not None and crop.size:
                ok, encoded = cv2.imencode(".jpg", crop)
                if not ok:
                    raise RuntimeError("Could not encode facial crop")
                with tempfile.NamedTemporaryFile(
                    dir=crop_dir, prefix=".pending-", delete=False
                ) as stream:
                    staged = Path(stream.name)
                    stream.write(encoded.tobytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                final = crop_dir / f"event_{uuid4().hex}.jpg"
            with database(db_path) as db:
                cursor = db.execute(
                    "INSERT INTO events(captured_at,status,profile_name,similarity,image_path,"
                    "track_id,session_id,model_id,episode_id,event_key) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        captured.astimezone(timezone.utc).isoformat(),
                        status,
                        name,
                        score,
                        str(final.resolve()) if final else None,
                        track_id,
                        session_id,
                        model_id,
                        episode_id,
                        event_key,
                    ),
                )
                event_id = int(cursor.lastrowid)
                if staged is not None:
                    staged.replace(final)
            return event_id
        except BaseException:
            # Normal failures roll back the row and clean staged/final files.
            # Power loss between rename and commit is handled by reconciliation.
            if final is not None:
                final.unlink(missing_ok=True)
            raise
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)


def maintain_events(
    db_path: Path,
    crop_dir: Path,
    retention_days: float | None = None,
    max_events: int | None = None,
) -> None:
    """Reconcile interrupted writes and optionally enforce retention/event count.

    All event writers cooperate with the same lock. Unreferenced generated crops
    are removed; arbitrary user files are left alone. Missing crops are marked NULL.
    """
    if retention_days is not None and (not math.isfinite(retention_days) or retention_days <= 0):
        raise ValueError("Retention days must be finite and positive")
    if max_events is not None and max_events <= 0:
        raise ValueError("Maximum event count must be positive")
    private_directory(crop_dir)
    with file_lock(db_path):
        with database(db_path) as db:
            if retention_days is not None:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
                db.execute("DELETE FROM events WHERE captured_at < ?", (cutoff,))
            if max_events is not None:
                db.execute(
                    "DELETE FROM events WHERE id NOT IN "
                    "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                    (max_events,),
                )
            referenced = set()
            for event_id, image in db.execute(
                "SELECT id,image_path FROM events WHERE image_path IS NOT NULL"
            ):
                path = Path(image)
                if path.is_file() and not path.is_symlink():
                    referenced.add(path.resolve())
                else:
                    db.execute("UPDATE events SET image_path=NULL WHERE id=?", (event_id,))
        # Delete only after database changes commit, so a failed commit cannot
        # remove an image still referenced by a retained event.
        for path in crop_dir.iterdir():
            generated = re.fullmatch(r"event_(?:[0-9a-f]{32}|[0-9]{6,})\.jpg", path.name)
            if (
                generated or path.name.startswith(".pending-")
            ) and path.resolve() not in referenced:
                path.unlink(missing_ok=True)


def save_episodes(
    db_path: Path, episodes: list, session_id: str, wall_start: datetime, monotonic_start: float
) -> None:
    if not episodes:
        return
    with file_lock(db_path), database(db_path) as db:
        for episode in episodes:
            started = wall_start + timedelta(seconds=episode.started - monotonic_start)
            seen = wall_start + timedelta(seconds=episode.last_seen - monotonic_start)
            db.execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?) ON CONFLICT(episode_id) "
                "DO UPDATE SET last_seen=excluded.last_seen,state=excluded.state",
                (
                    episode.episode_id,
                    session_id,
                    started.isoformat(),
                    seen.isoformat(),
                    episode.state,
                ),
            )
