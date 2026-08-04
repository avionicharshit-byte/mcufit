from pathlib import Path

import pytest

from mcufit.estimation.greedy import GreedyLifetimeEstimator
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture(scope="module")
def estimator() -> GreedyLifetimeEstimator:
    return GreedyLifetimeEstimator()


def test_tiny_model_has_tiny_arena(estimator):
    model = TFLiteModelParser().parse(MODELS / "hello_world_int8.tflite")
    estimate = estimator.estimate(model)
    assert estimate.peak_activation_bytes < 1024
    assert estimate.total_arena_bytes < 16 * 1024


def test_person_detect_arena_in_known_range(estimator):
    """The official TFLM person_detection example allocates a 96 KB arena
    for this model; a sane estimate must land in that neighborhood."""
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    estimate = estimator.estimate(model)
    assert 40 * 1024 < estimate.total_arena_bytes < 120 * 1024


def test_peak_layer_is_within_schedule(estimator):
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    estimate = estimator.estimate(model)
    assert 0 <= estimate.peak_layer_index < len(model.layers)


def test_margin_scales_with_peak(estimator):
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    estimate = estimator.estimate(model)
    assert estimate.margin_bytes == int(estimate.peak_activation_bytes * 0.20)
