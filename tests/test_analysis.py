from pathlib import Path

import pytest

from mcufit.analysis.latency import (
    count_macs,
    estimate_latency,
    macs_by_op,
    measured_boards,
)
from mcufit.analysis.quantization import project_int8, projected_file_size
from mcufit.boards.yaml_repo import YamlBoardRepository
from mcufit.domain.model import Quantization
from mcufit.estimation.greedy import GreedyLifetimeEstimator
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture(scope="module")
def person_detect():
    return TFLiteModelParser().parse(MODELS / "person_detect.tflite")


@pytest.fixture(scope="module")
def hello_float():
    return TFLiteModelParser().parse(MODELS / "hello_world_float.tflite")


def test_int8_projection_shrinks_float_model(hello_float):
    projected = project_int8(hello_float)
    assert projected.quantization == Quantization.INT8
    assert projected.weights_bytes < hello_float.weights_bytes
    original = GreedyLifetimeEstimator().estimate(hello_float)
    shrunk = GreedyLifetimeEstimator().estimate(projected)
    assert shrunk.peak_activation_bytes < original.peak_activation_bytes


def test_int8_projection_keeps_int_model_unchanged(person_detect):
    projected = project_int8(person_detect)
    assert projected.weights_bytes == person_detect.weights_bytes


def test_projected_file_size_shrinks_only_float(hello_float, person_detect):
    assert projected_file_size(hello_float) < hello_float.file_size_bytes
    assert projected_file_size(person_detect) == person_detect.file_size_bytes


def test_person_detect_macs_in_mobilenet_range(person_detect):
    # person_detect is MobileNet v1 0.25 @ 96x96: single-digit millions of MACs
    macs = count_macs(person_detect)
    assert 2_000_000 < macs < 20_000_000


def test_unmeasured_boards_get_no_latency(person_detect):
    boards = YamlBoardRepository()
    for board in boards.list():
        if board.id in measured_boards():
            continue
        assert estimate_latency(person_detect, board) is None, board.id


def test_measured_boards_reproduce_the_device(person_detect):
    # person_detect on real hardware: 474 ms on ESP32, 655 ms on nano33ble
    # (mcufit-bench, 2026-08-16). Bound covers the stated validation error.
    boards = YamlBoardRepository()
    for board_id, actual_ms in (("esp32", 474), ("nano33ble", 655)):
        est = estimate_latency(person_detect, boards.get(board_id))
        assert est is not None
        error = abs(est.milliseconds - actual_ms) / actual_ms
        assert error < 0.25, f"{board_id}: {est.milliseconds:.0f} vs {actual_ms}"


def test_macs_by_op_sums_to_total(person_detect):
    assert sum(macs_by_op(person_detect).values()) == count_macs(person_detect)
