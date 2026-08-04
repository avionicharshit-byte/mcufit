"""Loads boards from the bundled YAML database."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from ..domain.board import Board
from .base import UnknownBoardError


class YamlBoardRepository:
    """BoardRepository implementation backed by a YAML file.

    Defaults to the database shipped inside the package; a custom path may
    be supplied (e.g. a project-local boards file).
    """

    def __init__(self, source: Path | None = None):
        self._source = source
        self._boards: dict[str, Board] | None = None

    def get(self, board_id: str) -> Board:
        boards = self._load()
        key = board_id.strip().lower()
        if key not in boards:
            raise UnknownBoardError(key, sorted(boards))
        return boards[key]

    def list(self) -> list[Board]:
        return sorted(self._load().values(), key=lambda b: b.sram_bytes)

    def _load(self) -> dict[str, Board]:
        if self._boards is None:
            if self._source is not None:
                raw = yaml.safe_load(self._source.read_text())
            else:
                data = resources.files("mcufit.boards.data").joinpath("boards.yaml")
                raw = yaml.safe_load(data.read_text())
            self._boards = {}
            for entry in raw.get("boards", []):
                board = Board(
                    id=entry["id"],
                    name=entry["name"],
                    chip=entry["chip"],
                    sram_bytes=int(entry["sram"]),
                    flash_bytes=int(entry["flash"]),
                    reserved_sram_bytes=int(entry.get("reserved_sram", 0)),
                    psram_bytes=int(entry.get("psram", 0)),
                    notes=entry.get("notes", ""),
                )
                self._boards[board.id] = board
        return self._boards
