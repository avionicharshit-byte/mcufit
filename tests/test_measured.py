from pathlib import Path

import pytest

from mcufit.estimation.measured import (
    MeasuredArenaEstimator,
    MeasurementUnavailableError,
    find_benchmark_binary,
)
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"

BENCHMARK_OUTPUT = """
[[ Table ]]: Arena
        Arena   Bytes   % Arena
        Total | 89248 |   100.00
NonPersistent | 55296 |    61.96
   Persistent | 33952 |    38.04
"""


def test_parses_arena_table():
    estimate = MeasuredArenaEstimator._parse(BENCHMARK_OUTPUT)
    assert estimate.total_arena_bytes == 89248
    assert estimate.peak_activation_bytes == 55296
    assert estimate.overhead_bytes == 33952
    assert estimate.margin_bytes == 0
    assert estimate.method == "tflm-host-measurement"


def test_host_measurement_admits_it_is_an_upper_bound():
    # A host build is 64-bit, so its share of the arena spent on interpreter
    # bookkeeping is larger than a 32-bit device needs. Measured against a real
    # ESP32: 89,248 B here vs 82,300 B there. The number is allowed to be high,
    # it is not allowed to pretend it is the device's.
    estimate = MeasuredArenaEstimator._parse(BENCHMARK_OUTPUT)
    assert estimate.caveat is not None
    assert "upper bound" in estimate.caveat


def test_static_estimate_carries_no_caveat():
    # The caveat means something only if unqualified numbers stay unqualified.
    from mcufit.estimation.greedy import GreedyLifetimeEstimator

    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    assert GreedyLifetimeEstimator().estimate(model).caveat is None


def test_parse_failure_raises():
    with pytest.raises(MeasurementUnavailableError):
        MeasuredArenaEstimator._parse("no table here")


needs_binary = pytest.mark.skipif(
    find_benchmark_binary() is None,
    reason="TFLM benchmark binary not built (run `mcufit setup-exact`)",
)


@needs_binary
def test_measures_person_detect_exactly():
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    estimate = MeasuredArenaEstimator(binary=find_benchmark_binary()).estimate(model)
    # Exact for this model + TFLM version; loosely bounded so a TFLM update
    # shifts the number without breaking the suite.
    assert 60_000 < estimate.total_arena_bytes < 120_000
    assert estimate.margin_bytes == 0


@needs_binary
def test_measured_is_at_least_static_peak():
    from mcufit.estimation.greedy import GreedyLifetimeEstimator

    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    measured = MeasuredArenaEstimator(binary=find_benchmark_binary()).estimate(model)
    static = GreedyLifetimeEstimator().estimate(model)
    assert measured.total_arena_bytes >= static.peak_activation_bytes
