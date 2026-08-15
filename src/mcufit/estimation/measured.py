"""Arena measurement via the TFLM generic benchmark binary.

Runs the model through the real TFLite Micro interpreter compiled for this
machine and reads the recorded allocations, without flashing anything.
Requires a one-time `mcufit setup-exact` to clone and build tflite-micro.

Only half of that is portable, and this used to claim otherwise.

The arena has two sections. NonPersistent holds activation tensors, whose
sizes come from the model and are the same everywhere. Persistent holds the
interpreter's own bookkeeping, which is full of pointers, so a 64-bit host
build needs more of it than a 32-bit device does.

Measured on an ESP32-D0WDQ6 at 240 MHz, person_detect.tflite, against both of
mcufit's own builds (see avionicharshit-byte/mcufit-bench, 2026-08-16):

    section         host (arm64)   wasm32   device (esp32)
    NonPersistent         55,296   55,296           55,296   same everywhere
    Persistent            33,952   29,132           27,004
    Total                 89,248   84,428           82,300
    error vs device        +8.4%    +2.6%                -

Pointer width is the whole story: the 64-bit host needs 6,948 B more
bookkeeping than the device, and dropping to 32-bit recovers most of that.
What is left over in wasm32 is struct padding that differs from Xtensa.

Both over-report, which at least errs towards "needs more RAM than it does"
rather than the other way. Neither is the device's number. The fix is for
`--exact` to run the wasm32 build that `web/wasm/` already ships instead of
this native one, which would cut the error by three.
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
            caveat=(
                "activations are exact; the interpreter overhead is a 64-bit host "
                "figure and runs high on a 32-bit device (8.4% high in total on the "
                "one board measured so far), so treat the total as an upper bound. "
                "The browser version, built for 32-bit, comes within 2.6%"
            ),
        )


def _search(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None
