import json
import numpy as np
import pytest
from security_ai.diagnostics import diagnose
from security_ai import cli
from security_ai.capture import CapturedFrame
from security_ai.storage import data_paths, save_gallery
from test_security_integration import monitor_args
from test_pipeline import Analyzer, Detector
from unittest.mock import Mock


def test_offline_diagnostics_never_opens_camera(monkeypatch, capsys):
    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda *_: pytest.fail("camera opened"))
    report = diagnose()
    assert report["ready"] and not report["models_checked"]
    assert report["hardware_acceptance"].startswith("pending")
    monkeypatch.setattr("sys.argv", ["security-ai", "doctor"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["ready"]
    with pytest.raises(ValueError, match="require"):
        diagnose(check_models=True)


def test_stale_frames_respect_frame_limit(tmp_path, monkeypatch):
    gallery, db, crops = data_paths(tmp_path / "data")
    save_gallery(gallery, {"alice": [np.array([1.0, 0.0])]}, "test")
    model = Analyzer()
    model.fingerprint = "test"
    monkeypatch.setattr(cli, "create_analyzer", lambda *_: model)
    monkeypatch.setattr(cli, "YoloPersonDetector", lambda *_a, **_k: Detector())
    stream = Mock(dropped=0, reconnects=0)
    stream.start.return_value = stream
    stream.read.return_value = CapturedFrame(np.zeros((240, 320, 3), np.uint8), 0, 1)
    monkeypatch.setattr(cli, "LatestFrameCapture", lambda *_a, **_k: stream)
    args = monitor_args()
    args.max_frames = 3
    cli.monitor(args, gallery, db, crops)
    assert stream.read.call_count == 3
    stream.close.assert_called_once()
