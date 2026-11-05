"""Local SQLite events and NumPy enrollment gallery. NOT encrypted at rest."""

from __future__ import annotations

import re
from .core import GalleryIndex
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np


def data_paths(base: Path) -> tuple[Path, Path, Path]:
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base / "gallery.npz", base / "events.db", base / "crops"


def load_gallery(path: Path, expected_model: str | None = None) -> dict[str, list[np.ndarray]]:
    if not path.exists():
        return {}
    # allow_pickle=False is important: do not load executable pickle payloads.
    with np.load(path, allow_pickle=False) as archive:
        model = str(archive["__model__"]) if "__model__" in archive else "legacy"
        if expected_model is not None and model != expected_model:
            raise ValueError("Gallery model mismatch or legacy gallery; re-enroll profiles")
        gallery = {}
        for key in archive.files:
            if key == "__model__":
                continue
            safe_profile_name(key)
            values = archive[key]
            if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
                raise ValueError("Gallery entries must be nonempty sample matrices")
            gallery[key] = [np.asarray(row, dtype=np.float32) for row in values]
        GalleryIndex(gallery)
        return gallery


def save_gallery(
    path: Path, gallery: dict[str, list[np.ndarray]], model_id: str = "legacy"
) -> None:
    GalleryIndex(gallery)
    for name in gallery:
        safe_profile_name(name)
    arrays = {name: np.stack(samples).astype(np.float32) for name, samples in gallery.items()}
    arrays["__model__"] = np.asarray(model_id)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def safe_profile_name(name: str) -> str:
    if name == "__model__" or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("Profile names must use 1–40 ASCII letters, numbers, _ or -")
    return name


def init_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            status TEXT NOT NULL,
            profile_name TEXT,
            similarity REAL,
            image_path TEXT
        )""")


def save_event(
    db_path: Path,
    crop_dir: Path,
    status: str,
    name: str | None,
    score: float,
    crop: np.ndarray | None,
) -> int:
    captured = datetime.now(timezone.utc).isoformat()
    crop_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with sqlite3.connect(db_path) as db:
        cursor = db.execute(
            "INSERT INTO events(captured_at,status,profile_name,similarity,image_path) "
            "VALUES (?,?,?,?,NULL)",
            (captured, status, name, score),
        )
        event_id = int(cursor.lastrowid)
        if crop is not None and crop.size:
            path = crop_dir / f"event_{event_id:06d}.jpg"
            if not cv2.imwrite(str(path), crop):
                raise RuntimeError("Could not write facial crop")
            db.execute("UPDATE events SET image_path=? WHERE id=?", (str(path), event_id))
    return event_id
