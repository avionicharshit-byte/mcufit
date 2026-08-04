"""Machine-readable output for CI pipelines and scripting."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from ..domain.report import FitReport


class JsonReportRenderer:
    """ReportRenderer implementation emitting a single JSON document."""

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stdout

    def render(self, report: FitReport) -> None:
        est = report.estimate
        payload = {
            "model": {
                "path": str(report.model.path),
                "file_size_bytes": report.model.file_size_bytes,
                "quantization": report.model.quantization.value,
                "layers": len(report.model.layers),
                "weights_bytes": report.model.weights_bytes,
            },
            "board": {
                "id": report.board.id,
                "name": report.board.name,
                "sram_bytes": report.board.sram_bytes,
                "usable_sram_bytes": report.board.usable_sram_bytes,
                "flash_bytes": report.board.flash_bytes,
            },
            "estimate": {
                "peak_activation_bytes": est.peak_activation_bytes,
                "overhead_bytes": est.overhead_bytes,
                "margin_bytes": est.margin_bytes,
                "total_arena_bytes": est.total_arena_bytes,
                "peak_layer_index": est.peak_layer_index,
                "method": est.method,
            },
            "flash_needed_bytes": report.flash_needed_bytes,
            "fits": report.fits,
            "fits_ram": report.fits_ram,
            "fits_flash": report.fits_flash,
            "ram_utilization": round(report.ram_utilization, 4),
            "flash_utilization": round(report.flash_utilization, 4),
            "suggestions": [s.text for s in report.suggestions],
        }
        json.dump(payload, self._stream, indent=2)
        self._stream.write("\n")
