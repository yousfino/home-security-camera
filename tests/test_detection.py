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


def test_empty_detections_and_rgb_input(tmp_path):
    path = tmp_path / "fake.onnx"
    path.touch()
    session = Session()
    frame = np.zeros((640, 640, 3), np.uint8)
    frame[:, :, 2] = 255
    original = session.run

    def empty_run(outputs, inputs):
        original(outputs, inputs)
        return [np.zeros((1, 84, 10), np.float32)]

    session.run = empty_run
    assert YoloPersonDetector(path, session=session).detect(frame) == []
    assert session.tensor[0, 0, 0, 0] == 1 and session.tensor[0, 2, 0, 0] == 0


def test_non_float32_model_rejected(tmp_path):
    path = tmp_path / "fake.onnx"
    path.touch()
    session = Session()
    session.get_inputs = lambda: [
        SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float16)")
    ]
    with pytest.raises(ValueError, match="float32"):
        YoloPersonDetector(path, session=session)


def test_onnxruntime_cpu_executes_synthetic_export(tmp_path):
    # An executable constant-output graph validates the real ORT boundary without
    # trained weights, a download, a webcam or any captured image.
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    output = np.zeros((1, 84, 2), np.float32)
    output[0, :4, 0] = [320, 320, 200, 200]
    output[0, 4, 0] = 0.9
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))],
        "synthetic-yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "synthetic.onnx"
    onnx.save(model, path)
    detector = YoloPersonDetector(path)
    assert detector.session.get_providers() == ["CPUExecutionProvider"]
    result = detector.detect(np.zeros((640, 640, 3), np.uint8))
    assert len(result) == 1 and result[0].box == pytest.approx((220, 220, 420, 420))
