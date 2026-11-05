"""InsightFace model loading and basic quality checks."""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import numpy as np
from .quality import QualityConfig, assess_quality


class LocalFaceAnalyzer:
    """Detect first; run ArcFace only after association and quality checks."""

    def __init__(self, analyzer: object, fingerprint: str) -> None:
        self.analyzer = analyzer
        self.fingerprint = fingerprint

    def detect(self, frame: np.ndarray) -> list[object]:
        from insightface.app.common import Face

        boxes, landmarks = self.analyzer.det_model.detect(frame, max_num=0, metric="default")
        return [
            Face(bbox=box[:4], det_score=box[4], kps=None if landmarks is None else landmarks[i])
            for i, box in enumerate(boxes)
        ]

    def embed(self, frame: np.ndarray, face: object) -> np.ndarray:
        self.analyzer.models["recognition"].get(frame, face)
        return face.embedding

    def get(self, frame: np.ndarray) -> list[object]:
        faces = self.detect(frame)
        for face in faces:
            self.embed(frame, face)
        return faces


def create_analyzer(det_size: int = 640, root: Path | None = None) -> LocalFaceAnalyzer:
    root = (root or Path("~/.insightface")).expanduser()
    directory = root / "models" / "buffalo_l"
    files = sorted(directory.glob("*.onnx"))
    if not files:
        raise ValueError(
            f"Provision licensed buffalo_l ONNX weights in {directory}; no automatic downloads"
        )
    # InsightFace imports Albumentations, whose version checker otherwise performs
    # network requests on import. Runtime inference must stay local.
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(
        name="buffalo_l",
        root=str(root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    if "recognition" not in analyzer.models:
        raise ValueError("Model pack must contain detection and recognition models")
    analyzer.prepare(ctx_id=-1, det_size=(det_size, det_size))
    recognition = Path(analyzer.models["recognition"].model_file)
    with recognition.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    return LocalFaceAnalyzer(analyzer, fingerprint)


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
    frame: np.ndarray,
    face: object,
    policy: QualityConfig | None = None,
    min_face_px: int = 80,
    min_blur_variance: float = 55.0,
) -> bool:
    return assess_quality(
        frame,
        face,
        policy or QualityConfig(min_face_px=min_face_px, min_blur_variance=min_blur_variance),
    )
