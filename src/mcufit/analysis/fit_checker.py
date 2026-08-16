"""Turns a parsed model + target board into a fit verdict with suggestions."""

from __future__ import annotations

from ..boards.base import BoardRepository
from ..domain.board import Board
from ..domain.model import ModelInfo, Quantization
from ..domain.report import FitReport, Suggestion
from ..estimation.base import ArenaEstimator
from .quantization import project_int8, projected_file_size

# Flash consumed by the TFLite Micro runtime and kernels, on top of the model.
#
# Measured 2026-08-16 by compiling an empty sketch and diffing against the
# benchmark firmware, so the core and OS are excluded from this figure:
#
#   Nano 33 BLE (Cortex-M4F, mbed core, CMSIS-NN)     84,528 B
#   Arduino Nano ESP32 (Xtensa LX7, esp32 core, esp-nn) 114,579 B
#
# The larger is used, because between two measurements the conservative choice
# for a fit check is the one that needs more room. This replaces a hardcoded
# 150 KB that nothing had ever validated.
_TFLM_RUNTIME_FLASH_BYTES = 114_579

# What this number is NOT. The application, RTOS, radio stack and bootloader all
# share the same flash and mcufit cannot see any of them: an empty sketch alone
# costs 85 KB on the Nano 33 BLE and 347 KB on the Nano ESP32. So the flash
# figure is a **floor**, not a total. A "will not fit" verdict on flash is
# therefore certain, while a "fits" only means the model and runtime fit and
# says how much room is left for everything else.
_FLASH_IS_A_FLOOR = (
    "Flash counts the model and the TFLM runtime only. Your application, RTOS "
    "and radio stack are not included, and an empty sketch alone costs 85 KB on "
    "a Nano 33 BLE and 347 KB on an ESP32-S3."
)


class FitChecker:
    def __init__(self, estimator: ArenaEstimator, boards: BoardRepository):
        self._estimator = estimator
        self._boards = boards
        self._cached: tuple[ModelInfo, object] | None = None

    def _estimate(self, model: ModelInfo):
        """Estimate once per model.

        The arena depends on the model and the runtime, never on the board: a
        real ESP32 and a real nRF52840 came out 82,300 and 82,740 B on the same
        model. `check_all` used to recompute it for all 31 boards, which cost 31
        subprocess launches for one identical answer once measuring became the
        default.
        """
        if self._cached is None or self._cached[0] is not model:
            self._cached = (model, self._estimator.estimate(model))
        return self._cached[1]

    def check(self, model: ModelInfo, board: Board) -> FitReport:
        estimate = self._estimate(model)
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
            flash_left = report.board.flash_bytes - report.flash_needed_bytes
            yield Suggestion(
                f"Leaves ~{_fmt(headroom)} RAM for your application, "
                "sensor buffers, and network stack."
            )
            # The flash figure is a floor, so say what is left rather than
            # letting a green bar imply the whole firmware was accounted for.
            yield Suggestion(
                f"Leaves ~{_fmt(flash_left)} flash. {_FLASH_IS_A_FLOOR}"
            )
            return

        if report.fits_ram and not report.fits_flash:
            yield Suggestion(
                f"Model + runtime need {_fmt(report.flash_needed_bytes)} flash but the "
                f"board has {_fmt(report.board.flash_bytes)}. Prune or quantize the model."
            )

        if not report.fits_ram:
            if report.model.quantization == Quantization.FLOAT32:
                int8_arena = self._int8_projection(report.model)
                verdict = "fits" if int8_arena <= report.board.usable_sram_bytes else "still does not fit"
                yield Suggestion(
                    f"Quantize float32 -> int8: est. arena drops to ~{_fmt(int8_arena)}, "
                    f"file to ~{_fmt(projected_file_size(report.model))} "
                    f"({verdict} on this board)."
                )
            if report.board.psram_bytes:
                yield Suggestion(
                    f"This board offers {_fmt(report.board.psram_bytes)} PSRAM - placing the "
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
    def _int8_projection(model: ModelInfo) -> int:
        # Re-run the lifetime analysis on the int8-transformed graph rather
        # than dividing totals by four - bias tensors and alignment don't
        # shrink, and this accounts for both.
        from ..estimation.greedy import GreedyLifetimeEstimator

        return GreedyLifetimeEstimator().estimate(project_int8(model)).total_arena_bytes


def _fmt(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.0f} KB"
