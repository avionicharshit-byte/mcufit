"""Domain object describing a target microcontroller board."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Board:
    id: str
    name: str
    chip: str
    sram_bytes: int
    flash_bytes: int
    reserved_sram_bytes: int = 0
    """RAM the RTOS/radio stack eats before the app gets any — verdicts
    should reflect reality, not the datasheet."""
    psram_bytes: int = 0
    notes: str = ""

    @property
    def usable_sram_bytes(self) -> int:
        return max(0, self.sram_bytes - self.reserved_sram_bytes)
