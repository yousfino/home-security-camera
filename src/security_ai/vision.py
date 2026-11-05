"""InsightFace model loading and basic quality checks."""

from __future__ import annotations

import cv2
import numpy as np


def create_analyzer(det_size: int = 640):
    # Lazy import permits pure unit tests without loading native ML libraries.
    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return analyzer


def face_box(face, frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def crop_face(frame: np.ndarray, face, margin: float = 0.15) -> np.ndarray:
    x1, y1, x2, y2 = face_box(face, frame)
    h, w = frame.shape[:2]
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return frame[max(0, y1 - dy) : min(h, y2 + dy), max(0, x1 - dx) : min(w, x2 + dx)].copy()


def quality_ok(
    frame: np.ndarray, face, min_face_px: int = 80, min_blur_variance: float = 55.0
) -> bool:
    x1, y1, x2, y2 = face_box(face, frame)
    if x2 - x1 < min_face_px or y2 - y1 < min_face_px:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= min_blur_variance
