"""The core service: combines a parsed model, an estimator, and a board
into a fit verdict with actionable suggestions.

Depends only on the ArenaEstimator and BoardRepository abstractions
(dependency inversion) — concrete implementations are wired in the CLI.
"""

from __future__ import annotations

from ..boards.base import BoardRepository
from ..domain.board import Board
from ..domain.model import ModelInfo, Quantization
from ..domain.report import FitReport, MemoryEstimate, Suggestion
from ..estimation.base import ArenaEstimator

# Flash consumed by the TFLite Micro runtime + kernels themselves, on top
# of the model file. Varies with the op set linked in; this is a typical
# figure for a small int8 CNN build on Cortex-M/Xtensa.
_TFLM_RUNTIME_FLASH_BYTES = 150 * 1024

_INT8_SHRINK_FACTOR = 4  # float32 -> int8 shrinks weights and activations ~4x


class FitChecker:
    def __init__(self, estimator: ArenaEstimator, boards: BoardRepository):
        self._estimator = estimator
        self._boards = boards

    def check(self, model: ModelInfo, board: Board) -> FitReport:
        estimate = self._estimator.estimate(model)
        flash_needed = model.file_size_bytes + _TFLM_RUNTIME_FLASH_BYTES
        report = FitReport(
            model=model,
            board=board,
            estimate=estimate,
            flash_needed_bytes=flash_needed,
        )
        suggestions = tuple(self._suggest(report))
        return FitReport(
            model=model,
            board=board,
            estimate=estimate,
            flash_needed_bytes=flash_needed,
            suggestions=suggestions,
        )

    def check_all(self, model: ModelInfo) -> list[FitReport]:
        return [self.check(model, board) for board in self._boards.list()]

    def _suggest(self, report: FitReport):
        if report.fits:
            headroom = report.board.usable_sram_bytes - report.estimate.total_arena_bytes
            yield Suggestion(
                f"Leaves ~{_fmt(headroom)} RAM for your application, "
                "sensor buffers, and network stack."
            )
            return

        if report.fits_ram and not report.fits_flash:
            yield Suggestion(
                f"Model + runtime need {_fmt(report.flash_needed_bytes)} flash but the "
                f"board has {_fmt(report.board.flash_bytes)}. Prune or quantize the model."
            )

        if not report.fits_ram:
            if report.model.quantization == Quantization.FLOAT32:
                int8_arena = self._int8_projection(report.estimate)
                verdict = "fits" if int8_arena <= report.board.usable_sram_bytes else "still does not fit"
                yield Suggestion(
                    f"Quantize float32 -> int8: est. arena drops to ~{_fmt(int8_arena)} "
                    f"({verdict} on this board)."
                )
            if report.board.psram_bytes:
                yield Suggestion(
                    f"This board offers {_fmt(report.board.psram_bytes)} PSRAM — placing the "
                    "arena there fits easily, at some latency cost."
                )

        fitting = [
            board
            for board in self._boards.list()
            if board.id != report.board.id
            and report.estimate.total_arena_bytes <= board.usable_sram_bytes
            and report.flash_needed_bytes <= board.flash_bytes
        ][:3]
        if fitting:
            names = ", ".join(b.name for b in fitting)
            yield Suggestion(f"Smallest boards that fit this model as-is: {names}.")

    @staticmethod
    def _int8_projection(estimate: MemoryEstimate) -> int:
        return (
            estimate.peak_activation_bytes // _INT8_SHRINK_FACTOR
            + estimate.overhead_bytes
            + estimate.margin_bytes // _INT8_SHRINK_FACTOR
        )


def _fmt(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.0f} KB"
