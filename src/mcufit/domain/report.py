"""Domain objects describing the outcome of a fit check."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board
from .model import ModelInfo


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated runtime RAM (tensor arena) requirement for a model."""

    peak_activation_bytes: int
    """Largest sum of simultaneously-live activation tensors."""
    overhead_bytes: int
    """Interpreter + tensor metadata overhead added on top of activations."""
    margin_bytes: int
    """Safety margin covering op scratch buffers the static analysis can't see."""
    peak_layer_index: int
    method: str

    @property
    def total_arena_bytes(self) -> int:
        return self.peak_activation_bytes + self.overhead_bytes + self.margin_bytes


@dataclass(frozen=True)
class Suggestion:
    text: str


@dataclass(frozen=True)
class FitReport:
    """The verdict: does this model fit on this board?"""

    model: ModelInfo
    board: Board
    estimate: MemoryEstimate
    flash_needed_bytes: int
    suggestions: tuple[Suggestion, ...] = field(default=())

    @property
    def fits_ram(self) -> bool:
        return self.estimate.total_arena_bytes <= self.board.usable_sram_bytes

    @property
    def fits_flash(self) -> bool:
        return self.flash_needed_bytes <= self.board.flash_bytes

    @property
    def fits(self) -> bool:
        return self.fits_ram and self.fits_flash

    @property
    def ram_utilization(self) -> float:
        if self.board.usable_sram_bytes == 0:
            return float("inf")
        return self.estimate.total_arena_bytes / self.board.usable_sram_bytes

    @property
    def flash_utilization(self) -> float:
        if self.board.flash_bytes == 0:
            return float("inf")
        return self.flash_needed_bytes / self.board.flash_bytes
