"""mcufit command-line interface. All concrete implementations are wired here."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .analysis.fit_checker import FitChecker
from .boards.base import UnknownBoardError
from .boards.yaml_repo import YamlBoardRepository
from .domain.model import ModelInfo
from .estimation.base import ArenaEstimator
from .estimation.greedy import GreedyLifetimeEstimator
from .estimation.measured import (
    MeasuredArenaEstimator,
    MeasurementUnavailableError,
    find_benchmark_binary,
)
from .parsing.base import ModelParseError
from .parsing.tflite_parser import TFLiteModelParser
from .reporting.json_renderer import JsonReportRenderer
from .reporting.rich_renderer import RichReportRenderer

app = typer.Typer(
    name="mcufit",
    help="Check if an AI model fits on a microcontroller — before you flash it.",
    no_args_is_help=True,
)
console = Console()


def _parse_model(model_path: Path) -> ModelInfo:
    from .parsing.onnx_parser import OnnxModelParser

    for parser in (TFLiteModelParser(), OnnxModelParser()):
        if parser.supports(model_path):
            try:
                return parser.parse(model_path)
            except ModelParseError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(2)
    console.print(f"[red]Unsupported model format: {model_path.suffix} (expected .tflite or .onnx)[/red]")
    raise typer.Exit(2)


def _estimator(exact: bool) -> ArenaEstimator:
    if not exact:
        return GreedyLifetimeEstimator()
    binary = find_benchmark_binary()
    if binary is None:
        console.print(
            "[red]Exact mode needs the TFLM benchmark binary.[/red] "
            "Run [bold]mcufit setup-exact[/bold] once to build it (~5 min)."
        )
        raise typer.Exit(2)
    return MeasuredArenaEstimator(binary=binary)


def _checker(exact: bool = False) -> FitChecker:
    return FitChecker(estimator=_estimator(exact), boards=YamlBoardRepository())


def _fmt(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


@app.command()
def check(
    model: Path = typer.Argument(..., exists=True, readable=True, help="Path to a .tflite model"),
    board: str = typer.Option(..., "--board", "-b", help="Target board id (see `mcufit boards`)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (for CI)"),
    exact: bool = typer.Option(False, "--exact", "-x", help="Measure with the real TFLM runtime (needs `mcufit setup-exact`)"),
):
    """Check whether MODEL fits on BOARD. Exit code 1 if it does not."""
    if exact and model.suffix.lower() == ".onnx":
        console.print("[red]Exact mode runs the TFLM runtime and supports .tflite only.[/red]")
        raise typer.Exit(2)
    model_info = _parse_model(model)
    checker = _checker(exact)
    try:
        target = YamlBoardRepository().get(board)
    except UnknownBoardError as exc:
        console.print(f"[red]Unknown board '{exc.board_id}'.[/red] Known boards: {', '.join(exc.known)}")
        raise typer.Exit(2)

    try:
        report = checker.check(model_info, target)
    except MeasurementUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    renderer = JsonReportRenderer() if json_output else RichReportRenderer(console)
    renderer.render(report)
    raise typer.Exit(0 if report.fits else 1)


@app.command()
def boards():
    """List all boards in the database."""
    table = Table(title="mcufit board database")
    table.add_column("vendor", style="magenta")
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("usable SRAM", justify="right")
    table.add_column("flash", justify="right")
    table.add_column("PSRAM", justify="right")
    ordered = sorted(
        YamlBoardRepository().list(),
        key=lambda b: (b.vendor == "Other", b.vendor.lower(), b.name.lower()),
    )
    for b in ordered:
        table.add_row(
            b.vendor, b.id, b.name,
            _fmt(b.usable_sram_bytes), _fmt(b.flash_bytes),
            _fmt(b.psram_bytes) if b.psram_bytes else "—",
        )
    console.print(table)


@app.command()
def inspect(
    model: Path = typer.Argument(..., exists=True, readable=True, help="Path to a .tflite model"),
):
    """Show a layer-by-layer memory profile of MODEL."""
    model_info = _parse_model(model)
    estimate = GreedyLifetimeEstimator().estimate(model_info)

    console.print(
        f"\n[bold]{model_info.path.name}[/bold]  "
        f"({model_info.quantization.value}, {_fmt(model_info.file_size_bytes)} file, "
        f"{_fmt(model_info.weights_bytes)} weights, {len(model_info.layers)} layers)\n"
    )

    table = Table(title="Execution schedule")
    table.add_column("#", justify="right")
    table.add_column("op")
    table.add_column("output shape", style="dim")
    table.add_column("output size", justify="right")
    tensors = {t.index: t for t in model_info.tensors}
    for layer in model_info.layers:
        out = tensors.get(layer.output_tensors[0]) if layer.output_tensors else None
        marker = " [red]← peak[/red]" if layer.index == estimate.peak_layer_index else ""
        table.add_row(
            str(layer.index),
            layer.op_name + marker,
            "×".join(map(str, out.shape)) if out else "?",
            _fmt(out.size_bytes) if out else "?",
        )
    console.print(table)
    console.print(
        f"\nEstimated arena: [bold]~{_fmt(estimate.total_arena_bytes)}[/bold] "
        f"(peak activations {_fmt(estimate.peak_activation_bytes)} at layer "
        f"{estimate.peak_layer_index}, +overhead, +margin)\n"
    )


@app.command()
def compare(
    model: Path = typer.Argument(..., exists=True, readable=True, help="Path to a .tflite model"),
    exact: bool = typer.Option(False, "--exact", "-x", help="Measure with the real TFLM runtime"),
):
    """Check MODEL against every board in the database."""
    model_info = _parse_model(model)
    reports = _checker(exact).check_all(model_info)

    table = Table(title=f"{model_info.path.name} — fit across boards")
    table.add_column("board", style="cyan")
    table.add_column("usable SRAM", justify="right")
    table.add_column("arena needed", justify="right")
    table.add_column("verdict")
    for r in reports:
        verdict = "[green]✅ fits[/green]" if r.fits else "[red]❌ no[/red]"
        table.add_row(
            r.board.id,
            _fmt(r.board.usable_sram_bytes),
            f"~{_fmt(r.estimate.total_arena_bytes)}",
            verdict,
        )
    console.print(table)


@app.command("setup-exact")
def setup_exact():
    """Build the TFLM runtime for exact measurement mode (one-time, ~5 min)."""
    from .estimation.tflm_build import SetupError, build_benchmark

    existing = find_benchmark_binary()
    if existing:
        console.print(f"[green]Already set up:[/green] {existing}")
        return

    log = Path.home() / ".cache" / "mcufit" / "build.log"
    console.print("Cloning and building tflite-micro — this takes a few minutes...")
    try:
        with console.status("compiling"):
            binary = build_benchmark(log=log)
    except SetupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(f"[green]Done:[/green] {binary}\nUse it with: mcufit check model.tflite -b esp32-s3 --exact")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
