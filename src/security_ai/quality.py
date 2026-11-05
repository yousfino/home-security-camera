"""Configurable image/landmark heuristics, not occlusion or liveness detection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_face_px: int = 80
    min_blur_variance: float = 55.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    max_roll: float = 25.0
    max_yaw_asymmetry: float = 0.6
    require_landmarks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.min_blur_variance,
            self.min_brightness,
            self.max_brightness,
            self.max_roll,
            self.max_yaw_asymmetry,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Quality thresholds must be finite")
        if self.min_face_px <= 0 or self.min_blur_variance < 0:
            raise ValueError("Require positive face size and nonnegative blur threshold")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Brightness range must be within [0,255]")
        if not 0 <= self.max_roll <= 90 or not 0 <= self.max_yaw_asymmetry <= 1:
            raise ValueError("Invalid landmark geometry thresholds")


def assess_quality(frame: np.ndarray, face: object, policy: QualityConfig) -> bool:
    box = np.asarray(face.bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    x1, y1, x2, y2 = map(int, box)
    if min(x2 - x1, y2 - y1) < policy.min_face_px:
        return False
    crop = frame[y1:y2, x1:x2]
    if not crop.size:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if not policy.min_brightness <= float(gray.mean()) <= policy.max_brightness:
        return False
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < policy.min_blur_variance:
        return False
    landmarks = getattr(face, "kps", None)
    if landmarks is None:
        return not policy.require_landmarks
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return False
    if (
        (points[:, 0] < x1).any()
        or (points[:, 0] > x2).any()
        or (points[:, 1] < y1).any()
        or (points[:, 1] > y2).any()
    ):
        return False
    left, right, nose = points[:3]
    eyes = right - left
    eye_distance = float(np.linalg.norm(eyes))
    if eye_distance < 1 or eyes[0] <= 0:
        return False
    roll = abs(math.degrees(math.atan2(eyes[1], eyes[0])))
    asymmetry = (
        abs(float(np.linalg.norm(nose - left) - np.linalg.norm(nose - right))) / eye_distance
    )
    # Misordered/collapsed landmarks cannot establish a usable alignment.
    if nose[1] <= min(left[1], right[1]) or min(points[3:, 1]) <= nose[1]:
        return False
    return roll <= policy.max_roll and asymmetry <= policy.max_yaw_asymmetry
