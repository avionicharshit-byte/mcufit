"""Exact arena measurement via the TFLM generic benchmark binary.

Runs the model through the real TFLite Micro interpreter compiled for this
machine and reads the recorded allocations — the same numbers the device
would report, without flashing anything. Requires a one-time
`mcufit setup-exact` to clone and build tflite-micro.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..domain.model import ModelInfo
from ..domain.report import MemoryEstimate

TFLM_CACHE = Path.home() / ".cache" / "mcufit" / "tflite-micro"


class MeasurementUnavailableError(Exception):
    """The TFLM benchmark binary is missing or failed to run."""


def find_benchmark_binary(cache: Path = TFLM_CACHE) -> Path | None:
    candidates = sorted(cache.glob("gen/*/bin/tflm_benchmark"))
    return candidates[0] if candidates else None


@dataclass(frozen=True)
class MeasuredArenaEstimator:
    """ArenaEstimator implementation backed by a host-compiled TFLM run."""

    binary: Path

    def estimate(self, model: ModelInfo) -> MemoryEstimate:
        try:
            proc = subprocess.run(
                [str(self.binary), str(model.path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except OSError as exc:
            raise MeasurementUnavailableError(f"cannot run {self.binary}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MeasurementUnavailableError("TFLM benchmark timed out") from exc
        if proc.returncode != 0:
            raise MeasurementUnavailableError(
                "TFLM benchmark failed on this model "
                f"(exit {proc.returncode}):\n{proc.stderr[-500:] or proc.stdout[-500:]}"
            )
        # TFLM's MicroPrintf writes to stderr on host builds.
        return self._parse(proc.stdout + proc.stderr)

    @staticmethod
    def _parse(output: str) -> MemoryEstimate:
        # The benchmark prints an arena table:
        #   [[ Table ]]: Arena
        #           Total | 89248 |   100.00
        #   NonPersistent | 55296 |    61.96   <- tensors (activations)
        #      Persistent | 33952 |    38.04   <- op buffers, metadata
        section = output.split("[[ Table ]]: Arena", 1)[-1]
        total = _search(section, r"Total\s*\|\s*(\d+)")
        nonpersistent = _search(section, r"NonPersistent\s*\|\s*(\d+)")
        persistent = _search(section, r"\bPersistent\s*\|\s*(\d+)")
        if total is None:
            raise MeasurementUnavailableError(
                "could not find the arena table in TFLM benchmark output; "
                "please report this at "
                "https://github.com/avionicharshit-byte/mcufit/issues"
            )
        activations = nonpersistent if nonpersistent is not None else total
        overhead = persistent if persistent is not None else total - activations
        return MemoryEstimate(
            peak_activation_bytes=activations,
            overhead_bytes=overhead,
            margin_bytes=0,
            peak_layer_index=-1,  # the benchmark reports totals, not per-layer peaks
            method="tflm-host-measurement",
        )


def _search(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None
