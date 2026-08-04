"""Renders real mcufit output to docs/demo.svg for the README.

Run after output-affecting changes: .venv/bin/python scripts/render_demo.py
"""

from pathlib import Path

from rich.console import Console

from mcufit.analysis.fit_checker import FitChecker
from mcufit.boards.yaml_repo import YamlBoardRepository
from mcufit.estimation.greedy import GreedyLifetimeEstimator
from mcufit.parsing.tflite_parser import TFLiteModelParser
from mcufit.reporting.rich_renderer import RichReportRenderer

ROOT = Path(__file__).parent.parent


def main() -> None:
    console = Console(record=True, width=82, force_terminal=True)
    boards = YamlBoardRepository()
    checker = FitChecker(estimator=GreedyLifetimeEstimator(), boards=boards)
    model = TFLiteModelParser().parse(ROOT / "examples" / "models" / "person_detect.tflite")
    renderer = RichReportRenderer(console)

    console.print("[bold]$ mcufit check person_detect.tflite -b esp32-s3[/bold]")
    renderer.render(checker.check(model, boards.get("esp32-s3")))
    console.print("\n[bold]$ mcufit check person_detect.tflite -b uno[/bold]")
    renderer.render(checker.check(model, boards.get("uno")))

    out = ROOT / "docs" / "demo.svg"
    out.parent.mkdir(exist_ok=True)
    out.write_text(console.export_svg(title="mcufit"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
