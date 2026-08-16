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
    assert estimate.margin_bytes == int(estimate.peak_activation_bytes * estimator.scratch_margin)


def test_static_estimate_clears_the_real_esp32(estimator):
    # person_detect needs 82,300 B on a real ESP32 (mcufit-bench, 2026-08-16).
    # At the old 0.20 margin this path returned 76,147, so mcufit would have
    # called it a fit on a board with 80 KB free. Reading high is fine here;
    # reading low is not.
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    assert estimator.estimate(model).total_arena_bytes >= 82_300
