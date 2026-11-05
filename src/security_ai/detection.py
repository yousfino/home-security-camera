"""Local YOLOv8/YOLO11 COCO ONNX detection; no automatic downloads.

Accepts a standard raw export with output [1,84,N] or [1,N,84].
End-to-end/NMS exports and YOLOv5 exports are deliberately rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class PersonDetection:
    box: tuple[float, float, float, float]
    confidence: float


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[PersonDetection]: ...


def verify_model(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise ValueError("Supply an existing local .onnx model; weights are never downloaded")
    if expected_sha256 is not None:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError("Model SHA256 mismatch")


class YoloPersonDetector:
    def __init__(
        self,
        model: Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        sha256: str | None = None,
        session: object | None = None,
    ) -> None:
        if not 0 < confidence <= 1 or not 0 < iou <= 1:
            raise ValueError("Confidence and NMS IoU must be in (0,1]")
        verify_model(model, sha256)
        self.session = (
            session
            if session is not None
            else ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("Model has no input")
        if getattr(inputs[0], "type", "tensor(float)") != "tensor(float)":
            raise ValueError("Export with float32 input")
        shape = inputs[0].shape
        if len(inputs) != 1 or len(shape) != 4 or shape[:2] != [1, 3]:
            raise ValueError("Expected a static batch-1 NCHW model")
        if not all(isinstance(v, int) and v > 0 for v in shape[2:]):
            raise ValueError("Export with static image dimensions")
        self.height, self.width = shape[2:]
        self.input_name = inputs[0].name
        self.confidence, self.iou = confidence, iou

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (self.width - rw) // 2, (self.height - rh) // 2
        padded = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        padded[top : top + rh, left : left + rw] = cv2.resize(frame, (rw, rh))
        tensor = (
            np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        raw = self.session.run(None, {self.input_name: tensor})[0]
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise ValueError("Expected raw COCO YOLO output [1,84,N]")
        rows = raw[0].T if raw.shape[1] == 84 else raw[0]
        if rows.shape[1] != 84:
            raise ValueError("Only raw YOLOv8/YOLO11 COCO exports (84 channels) are supported")
        # Class zero is person. Reject detections whose winning class is not person.
        keep = np.isfinite(rows).all(axis=1) & (rows[:, 4] >= self.confidence)
        keep &= rows[:, 4:].argmax(axis=1) == 0
        rows = rows[keep]
        boxes, scores = [], []
        for row in rows:
            cx, cy, bw, bh = row[:4]
            x1, y1 = max(0.0, (cx - bw / 2 - left) / scale), max(0.0, (cy - bh / 2 - top) / scale)
            x2, y2 = (
                min(float(w), (cx + bw / 2 - left) / scale),
                min(float(h), (cy + bh / 2 - top) / scale),
            )
            if x2 > x1 and y2 > y1:
                boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                scores.append(float(row[4]))
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)
        return [
            PersonDetection(
                (boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
                scores[i],
            )
            for i in np.asarray(indices).reshape(-1)
        ]
