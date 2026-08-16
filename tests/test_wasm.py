from pathlib import Path

import pytest

from mcufit.estimation.wasm import WasmArenaEstimator, node_executable
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"

needs_node = pytest.mark.skipif(node_executable() is None, reason="node not on PATH")


def test_wasm_assets_ship_with_the_package():
    from importlib import resources

    files = resources.files("mcufit.wasm")
    assert files.joinpath("tflm.js").is_file()
    assert files.joinpath("run_benchmark.mjs").is_file()


@needs_node
def test_measures_person_detect():
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    estimate = WasmArenaEstimator(node=node_executable()).estimate(model)
    # Loosely bounded so a TFLM update shifts the number without breaking CI.
    assert 60_000 < estimate.total_arena_bytes < 120_000
    assert estimate.method == "tflm-wasm32-measurement"
    assert estimate.margin_bytes == 0


@needs_node
def test_activations_match_the_static_peak():
    # Activation sizes come from the model, so both paths must agree on them.
    from mcufit.estimation.greedy import GreedyLifetimeEstimator

    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    wasm = WasmArenaEstimator(node=node_executable()).estimate(model)
    static = GreedyLifetimeEstimator().estimate(model)
    assert wasm.peak_activation_bytes == static.peak_activation_bytes


@needs_node
def test_beats_the_host_build_against_a_real_esp32():
    # Real ESP32-D0WDQ6, person_detect: 82,300 B (mcufit-bench, 2026-08-16).
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    wasm = WasmArenaEstimator(node=node_executable()).estimate(model)
    error = abs(wasm.total_arena_bytes - 82_300) / 82_300
    assert error < 0.05


@needs_node
def test_carries_a_caveat():
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    assert WasmArenaEstimator(node=node_executable()).estimate(model).caveat is not None


def test_caveat_states_the_overshoot_in_bytes():
    # It used to say "~3% high", measured on person_detect alone. The overshoot is
    # a fixed 2,128 B, so on the anomaly detector's 4.5 KB arena that same slop is
    # nearly half the number. Stating bytes is what makes it true at both sizes.
    from mcufit.estimation.wasm import ESP32_OVERSHOOT_BYTES, _caveat

    big, small = _caveat(84_428), _caveat(4_508)
    assert f"{ESP32_OVERSHOOT_BYTES:,} B" in big
    assert "3% of this arena" in big
    assert "47% of this arena" in small
