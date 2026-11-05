from types import SimpleNamespace
import numpy as np
import pytest
from security_ai.detection import YoloPersonDetector, verify_model


class Session:
    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def run(self, _, inputs):
        assert inputs["images"].shape == (1, 3, 640, 640)
        self.tensor = inputs["images"]
        rows = np.zeros((3, 84), dtype=np.float32)
        rows[:, :4] = [320, 320, 320, 320]
        rows[:, 4] = [0.9, 0.8, 0.1]
        rows[2, 5] = 0.95  # other class
        return [rows.T[None]]


def test_letterbox_mapping_and_nms(tmp_path):
    path = tmp_path / "fake.onnx"
    path.touch()
    result = YoloPersonDetector(path, session=Session()).detect(
        np.zeros((320, 640, 3), dtype=np.uint8)
    )
    assert len(result) == 1
    assert result[0].box == pytest.approx((160, 0, 480, 320))
    assert result[0].confidence == pytest.approx(0.9)


def test_no_download_and_hash_check(tmp_path):
    with pytest.raises(ValueError, match="local"):
        verify_model(tmp_path / "missing.onnx")
    path = tmp_path / "fake.onnx"
    path.touch()
    with pytest.raises(ValueError, match="SHA256"):
        verify_model(path, "wrong")


def test_unsupported_export(tmp_path):
    path = tmp_path / "fake.onnx"
    path.touch()
    session = Session()
    session.run = lambda *_: [np.zeros((1, 6, 100))]
    with pytest.raises(ValueError, match="84 channels"):
        YoloPersonDetector(path, session=session).detect(np.zeros((100, 100, 3), dtype=np.uint8))
