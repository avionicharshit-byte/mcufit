"""Machine-readable output for CI pipelines and scripting."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from ..analysis.latency import count_macs, estimate_latency, macs_by_op
from ..domain.report import FitReport


def _latency(report) -> dict:
    """Measured or nothing. An unmeasured board gets no number to misread."""
    est = estimate_latency(report.model, report.board)
    if est is None:
        return {"measured": False, "milliseconds": None,
                "note": "no hardware measurement for this board"}
    return {
        "measured": True,
        "milliseconds": round(est.milliseconds, 1),
        "measured_on": est.measured_on,
        "kernels": est.kernels,
        "error_pct": list(est.error_pct),
        "unmodelled_ops": list(est.unmodelled_ops),
    }


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
                "caveat": est.caveat,
            },
            "flash_needed_bytes": report.flash_needed_bytes,
            "macs": count_macs(report.model),
            "macs_by_op": macs_by_op(report.model),
            "latency": _latency(report),
            "fits": report.fits,
            "fits_ram": report.fits_ram,
            "fits_flash": report.fits_flash,
            "ram_utilization": round(report.ram_utilization, 4),
            "flash_utilization": round(report.flash_utilization, 4),
            "suggestions": [s.text for s in report.suggestions],
        }
        json.dump(payload, self._stream, indent=2)
        self._stream.write("\n")
