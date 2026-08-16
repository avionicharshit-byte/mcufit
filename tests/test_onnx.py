import pytest

onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402
import numpy as np  # noqa: E402

from mcufit.domain.model import Quantization  # noqa: E402
from mcufit.estimation.greedy import GreedyLifetimeEstimator  # noqa: E402
from mcufit.parsing.onnx_parser import OnnxModelParser  # noqa: E402


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    """input[1,64] -> Gemm(w[32,64]) -> Relu -> output[1,32]"""
    weights = numpy_helper.from_array(
        np.zeros((32, 64), dtype=np.float32), name="w"
    )
    bias = numpy_helper.from_array(np.zeros(32, dtype=np.float32), name="b")
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["x", "w", "b"], ["h"], transB=1),
            helper.make_node("Relu", ["h"], ["y"]),
        ],
        name="tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        initializer=[weights, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_supports_onnx_extension():
    from pathlib import Path

    parser = OnnxModelParser()
    assert parser.supports(Path("m.onnx"))
    assert not parser.supports(Path("m.tflite"))


def test_parses_tiny_model(tiny_onnx):
    model = OnnxModelParser().parse(tiny_onnx)
    assert model.quantization == Quantization.FLOAT32
    assert [layer.op_name for layer in model.layers] == ["GEMM", "RELU"]
    # w (32*64*4) + b (32*4)
    assert model.weights_bytes == 32 * 64 * 4 + 32 * 4
    assert model.graph_inputs and model.graph_outputs


def test_estimator_runs_on_onnx(tiny_onnx):
    model = OnnxModelParser().parse(tiny_onnx)
    estimate = GreedyLifetimeEstimator().estimate(model)
    # x(256B) + h(128B) + y(128B) at various lifetimes; peak under 2 KB
    assert 0 < estimate.peak_activation_bytes < 2048


def test_onnx_verdict_says_tflm_cannot_run_it(tiny_onnx):
    # The most confident-looking number mcufit prints is also its least certain:
    # TFLM cannot execute ONNX at all, so the verdict describes a .tflite that
    # does not exist yet. Nothing in the output said so until 2026-08-16.
    from mcufit.analysis.fit_checker import FitChecker
    from mcufit.boards.yaml_repo import YamlBoardRepository

    boards = YamlBoardRepository()
    checker = FitChecker(estimator=GreedyLifetimeEstimator(), boards=boards)
    report = checker.check(OnnxModelParser().parse(tiny_onnx), boards.get("esp32-s3"))

    text = " ".join(s.text for s in report.suggestions)
    assert "TFLite Micro cannot run it" in text
    assert "not what you would flash" in text  # the tiny fixture is float32


def test_tflite_verdict_carries_no_onnx_warning():
    from pathlib import Path

    from mcufit.analysis.fit_checker import FitChecker
    from mcufit.boards.yaml_repo import YamlBoardRepository
    from mcufit.parsing.tflite_parser import TFLiteModelParser

    models = Path(__file__).parent.parent / "examples" / "models"
    boards = YamlBoardRepository()
    checker = FitChecker(estimator=GreedyLifetimeEstimator(), boards=boards)
    report = checker.check(
        TFLiteModelParser().parse(models / "person_detect.tflite"), boards.get("esp32-s3")
    )
    assert "ONNX" not in " ".join(s.text for s in report.suggestions)
