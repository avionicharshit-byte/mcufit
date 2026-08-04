from pathlib import Path

import pytest

from mcufit.analysis.fit_checker import FitChecker
from mcufit.boards.base import UnknownBoardError
from mcufit.boards.yaml_repo import YamlBoardRepository
from mcufit.estimation.greedy import GreedyLifetimeEstimator
from mcufit.parsing.tflite_parser import TFLiteModelParser

MODELS = Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture(scope="module")
def checker() -> FitChecker:
    return FitChecker(estimator=GreedyLifetimeEstimator(), boards=YamlBoardRepository())


@pytest.fixture(scope="module")
def boards() -> YamlBoardRepository:
    return YamlBoardRepository()


def test_person_detect_fits_esp32s3(checker, boards):
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    report = checker.check(model, boards.get("esp32-s3"))
    assert report.fits


def test_person_detect_does_not_fit_uno(checker, boards):
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    report = checker.check(model, boards.get("uno"))
    assert not report.fits
    assert not report.fits_ram
    assert not report.fits_flash


def test_failing_report_suggests_alternative_boards(checker, boards):
    model = TFLiteModelParser().parse(MODELS / "person_detect.tflite")
    report = checker.check(model, boards.get("uno"))
    assert any("boards that fit" in s.text for s in report.suggestions)


def test_float_model_gets_quantization_suggestion(checker, boards):
    model = TFLiteModelParser().parse(MODELS / "hello_world_float.tflite")
    # hello_world is tiny and fits everywhere, so force the RAM failure path
    # with the most constrained board in the database.
    report = checker.check(model, boards.get("uno"))
    assert any("int8" in s.text for s in report.suggestions)


def test_unknown_board_raises(boards):
    with pytest.raises(UnknownBoardError):
        boards.get("board-that-does-not-exist")


def test_board_database_loads_and_is_sorted(boards):
    all_boards = boards.list()
    assert len(all_boards) >= 10
    srams = [b.sram_bytes for b in all_boards]
    assert srams == sorted(srams)
