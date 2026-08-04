"""Board database integrity — a malformed boards.yaml PR should fail CI."""

import pytest

from mcufit.boards.yaml_repo import BoardDataError, YamlBoardRepository


@pytest.fixture(scope="module")
def boards():
    return YamlBoardRepository().list()


def test_database_has_expected_size(boards):
    assert len(boards) >= 30


def test_ids_are_unique_and_well_formed(boards):
    ids = [b.id for b in boards]
    assert len(ids) == len(set(ids))
    for board_id in ids:
        assert board_id == board_id.lower()
        assert " " not in board_id


def test_every_board_has_sane_memory(boards):
    for b in boards:
        assert b.sram_bytes > 0, b.id
        assert b.flash_bytes > 0, b.id
        assert 0 <= b.reserved_sram_bytes < b.sram_bytes, b.id
        assert b.usable_sram_bytes > 0, b.id
        assert b.psram_bytes >= 0, b.id


def test_every_board_has_identity_fields(boards):
    for b in boards:
        assert b.name.strip(), b.id
        assert b.chip.strip(), b.id


def test_known_families_are_present(boards):
    ids = {b.id for b in boards}
    for expected in ("esp32", "esp32-s3", "rp2040", "stm32f411", "teensy41", "uno-r4"):
        assert expected in ids


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"id": "x", "name": "X", "chip": "c", "sram": 1024}, "missing 'flash'"),
        ({"id": "BAD ID", "name": "X", "chip": "c", "sram": 1024, "flash": 1024}, "lowercase"),
        ({"id": "x", "name": "X", "chip": "c", "sram": 0, "flash": 1024}, "positive"),
        ({"id": "x", "name": "X", "chip": "c", "sram": 1024, "flash": 1024, "reserved_sram": 2048}, "reserves more"),
    ],
)
def test_loader_rejects_bad_entries(tmp_path, entry, message):
    import yaml

    source = tmp_path / "boards.yaml"
    source.write_text(yaml.safe_dump({"boards": [entry]}))
    with pytest.raises(BoardDataError, match=message):
        YamlBoardRepository(source=source).list()


def test_loader_rejects_duplicate_ids(tmp_path):
    import yaml

    entry = {"id": "x", "name": "X", "chip": "c", "sram": 1024, "flash": 1024}
    source = tmp_path / "boards.yaml"
    source.write_text(yaml.safe_dump({"boards": [entry, dict(entry)]}))
    with pytest.raises(BoardDataError, match="duplicate"):
        YamlBoardRepository(source=source).list()
