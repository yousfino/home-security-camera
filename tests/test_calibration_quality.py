import json
from types import SimpleNamespace
import numpy as np
import pytest
from security_ai.calibration import evaluate, load_scores, wilson
from security_ai.core import GalleryIndex
from security_ai.quality import QualityConfig, assess_quality
from security_ai import cli


def test_calibration_denominators_intervals_and_no_automatic_policy_change():
    report = evaluate([(0.9, True)] * 20 + [(0.2, False)] * 400, [0.1, 0.5, 0.95])
    assert report["genuine_count"] == 20 and report["impostor_count"] == 400
    assert report["suggested_threshold"] == 0.5
    assert report["results"][0]["fmr"] == 1
    assert report["results"][2]["fnmr"] == 1
    assert not report["runtime_configuration_changed"]
    assert wilson(0, 400)[1] < 0.01
    assert evaluate([(0.9, True), (0.1, False)], [0.5])["suggested_threshold"] is None


@pytest.mark.parametrize(
    "rows,thresholds",
    [
        ([], [0.5]),
        ([(float("nan"), True), (0.1, False)], [0.5]),
        ([(0.9, True), (0.1, False)], [2]),
    ],
)
def test_invalid_calibration_rejected(rows, thresholds):
    with pytest.raises(ValueError):
        evaluate(rows, thresholds)


def test_calibration_cli_and_invalid_csv(tmp_path, monkeypatch, capsys):
    path = tmp_path / "synthetic.csv"
    path.write_text("score,same_identity\n0.9,1\n0.1,0\n")
    assert len(load_scores(path)) == 2
    monkeypatch.setattr(
        "sys.argv",
        [
            "security-ai",
            "calibrate",
            "--scores",
            str(path),
            "--thresholds",
            ".5",
            "--enrollment-independent",
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["suggested_threshold"] is None
    path.write_text("score,same_identity\nnan,1\n.1,0\n")
    with pytest.raises(ValueError):
        load_scores(path)


def test_mean_identity_aggregation_can_reduce_single_sample_outlier():
    gallery = {"a": [np.array([1.0, 0.0]), np.array([0.0, 1.0])], "b": [np.array([0.8, 0.6])]}
    assert GalleryIndex(gallery).match(np.array([1.0, 0.0]))[0] == "a"
    assert GalleryIndex(gallery, "mean").match(np.array([1.0, 0.0]))[0] == "b"


def quality_fixture():
    grid = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    frame = np.repeat(grid[:, :, None], 3, axis=2)
    face = SimpleNamespace(
        bbox=np.array([10, 10, 90, 90]),
        kps=np.array([[30, 35], [70, 35], [50, 55], [35, 75], [65, 75]], float),
    )
    return frame, face


def test_exposure_blur_landmarks_and_geometry():
    frame, face = quality_fixture()
    assert assess_quality(frame, face, QualityConfig())
    assert not assess_quality(np.zeros_like(frame), face, QualityConfig())
    assert not assess_quality(np.full_like(frame, 255), face, QualityConfig())
    assert not assess_quality(np.full_like(frame, 127), face, QualityConfig())
    face.kps = None
    assert not assess_quality(frame, face, QualityConfig())
    assert assess_quality(frame, face, QualityConfig(require_landmarks=False))
    _, face = quality_fixture()
    face.kps[1, 1] = 65
    assert not assess_quality(frame, face, QualityConfig())
    _, face = quality_fixture()
    face.kps[2, 0] = 85
    assert not assess_quality(frame, face, QualityConfig(max_yaw_asymmetry=0.1))
    face.bbox = np.array([0, 0, float("nan"), 90])
    assert not assess_quality(frame, face, QualityConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_brightness": 230},
        {"max_roll": -1},
        {"min_face_px": 0},
        {"min_blur_variance": float("nan")},
    ],
)
def test_quality_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        QualityConfig(**kwargs)
