"""Board repository abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.board import Board


class UnknownBoardError(Exception):
    def __init__(self, board_id: str, known: list[str]):
        self.board_id = board_id
        self.known = known
        super().__init__(f"unknown board '{board_id}'")


@runtime_checkable
class BoardRepository(Protocol):
    def get(self, board_id: str) -> Board:
        """Look up a board by id. Raises UnknownBoardError."""
        ...

    def list(self) -> list[Board]:
        """All known boards."""
        ...
