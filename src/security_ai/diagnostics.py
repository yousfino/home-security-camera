"""Offline runtime diagnostics; never opens camera hardware."""

from __future__ import annotations

import ctypes
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import onnxruntime as ort


def diagnose(
    person_model: Path | None = None,
    face_root: Path | None = None,
    person_sha256: str | None = None,
    check_models: bool = False,
) -> dict:
    checks: dict[str, bool] = {
        "python_311": sys.version_info[:2] == (3, 11),
        "cpu_provider": "CPUExecutionProvider" in ort.get_available_providers(),
    }
    if sys.platform.startswith("linux"):
        for name in ("libGL.so.1", "libglib-2.0.so.0", "libgomp.so.1", "libstdc++.so.6"):
            try:
                ctypes.CDLL(name)
            except OSError:
                checks[name] = False
            else:
                checks[name] = True
    if check_models:
        if person_model is None or face_root is None:
            raise ValueError("Model checks require --person-model and --face-model-root")
        from .detection import YoloPersonDetector
        from .vision import create_analyzer

        YoloPersonDetector(person_model, sha256=person_sha256)
        create_analyzer(root=face_root)
        checks["local_models_load"] = True
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "opencv-python", "onnxruntime", "insightface", "supervision")
        },
        "models_checked": check_models,
        "hardware_acceptance": "pending: live camera, accuracy, crossings and soak test",
    }
