"""Pretty terminal output for fit reports."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..domain.report import FitReport

_BAR_WIDTH = 20


def _fmt(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def _bar(ratio: float) -> str:
    filled = min(_BAR_WIDTH, round(min(ratio, 1.0) * _BAR_WIDTH))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


class RichReportRenderer:
    """ReportRenderer implementation for interactive terminals."""

    def __init__(self, console: Console | None = None):
        self._console = console or Console()

    def render(self, report: FitReport) -> None:
        c = self._console
        model, board, est = report.model, report.board, report.estimate

        body = Text()
        body.append("Model:  ", style="bold")
        body.append(
            f"{model.path.name}  ({model.quantization.value}, "
            f"{len(model.layers)} layers, {_fmt(model.file_size_bytes)})\n"
        )
        body.append("Board:  ", style="bold")
        body.append(f"{board.name}  ({_fmt(board.usable_sram_bytes)} usable SRAM · {_fmt(board.flash_bytes)} flash)\n\n")

        if report.fits:
            body.append("✅ FITS\n\n", style="bold green")
        else:
            body.append("❌ WON'T FIT", style="bold red")
            reasons = []
            if not report.fits_ram:
                reasons.append(
                    f"needs ~{_fmt(est.total_arena_bytes)} RAM, "
                    f"board has {_fmt(board.usable_sram_bytes)} usable"
                )
            if not report.fits_flash:
                reasons.append(
                    f"needs {_fmt(report.flash_needed_bytes)} flash, "
                    f"board has {_fmt(board.flash_bytes)}"
                )
            body.append(f" — {'; '.join(reasons)}\n\n", style="red")

        ram_style = "green" if report.fits_ram else "red"
        body.append("RAM   ", style="bold")
        body.append(_bar(report.ram_utilization), style=ram_style)
        body.append(
            f"  ~{_fmt(est.total_arena_bytes)} arena / {_fmt(board.usable_sram_bytes)}"
            f"  ({min(report.ram_utilization, 9.99):.0%})\n"
        )
        flash_style = "green" if report.fits_flash else "red"
        body.append("Flash ", style="bold")
        body.append(_bar(report.flash_utilization), style=flash_style)
        body.append(
            f"  {_fmt(report.flash_needed_bytes)} total / {_fmt(board.flash_bytes)}"
            f"  ({min(report.flash_utilization, 9.99):.0%})\n\n"
        )

        if model.layers and est.peak_layer_index >= 0:
            peak = model.layers[est.peak_layer_index]
            body.append(
                f"Peak memory moment: layer {peak.index} ({peak.op_name}) — "
                f"{_fmt(est.peak_activation_bytes)} live tensors\n",
                style="dim",
            )
        body.append(
            f"Method: {est.method} (+{_fmt(est.overhead_bytes)} runtime overhead, "
            f"+{_fmt(est.margin_bytes)} scratch margin)\n",
            style="dim",
        )

        for suggestion in report.suggestions:
            body.append(f"\n • {suggestion.text}", style="cyan")

        title = "mcufit — does it fit?"
        c.print(Panel(body, title=title, border_style="green" if report.fits else "red"))
