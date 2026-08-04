from pathlib import Path

import pytest

from mcufit.domain.model import Quantization
from mcufit.parsing.base import ModelParseError
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture(scope="module")
def parser() -> TFLiteModelParser:
    return TFLiteModelParser()


def test_supports_tflite_extension(parser):
    assert parser.supports(Path("model.tflite"))
    assert not parser.supports(Path("model.onnx"))


def test_parses_int8_model(parser):
    model = parser.parse(MODELS / "hello_world_int8.tflite")
    assert model.quantization == Quantization.INT8
    assert len(model.layers) == 3
    assert all(layer.op_name == "FULLY_CONNECTED" for layer in model.layers)
    assert model.weights_bytes > 0
    assert model.graph_inputs and model.graph_outputs


def test_parses_float_model(parser):
    model = parser.parse(MODELS / "hello_world_float.tflite")
    assert model.quantization == Quantization.FLOAT32


def test_weights_vs_activations_split(parser):
    model = parser.parse(MODELS / "person_detect.tflite")
    weights = [t for t in model.tensors if t.is_weight]
    activations = model.activation_tensors
    assert weights and activations
    # person_detect is ~300 KB of int8 weights
    assert 200_000 < model.weights_bytes < 350_000


def test_rejects_garbage_file(parser, tmp_path):
    bad = tmp_path / "bad.tflite"
    bad.write_bytes(b"not a flatbuffer")
    with pytest.raises(ModelParseError):
        parser.parse(bad)
