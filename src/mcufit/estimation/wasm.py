"""Arena measurement via TFLM compiled to wasm32, run under node.

wasm32 is 32-bit like the target, so interpreter overhead lands close to a real
device instead of the host build's ~8% over. Needs node on PATH and nothing
else: the wasm is bundled in the wheel, so there is no `setup-exact` step.

The overshoot is a fixed number of bytes, not a percentage. See
ESP32_OVERSHOOT_BYTES below.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..domain.model import ModelInfo
from ..domain.report import MemoryEstimate
from .measured import MeasurementUnavailableError, parse_arena_table

SHIM = "run_benchmark.mjs"

# Measured 2026-08-16 against a real ESP32 across all four mcufit-bench models:
# the wasm32 arena came out exactly 2,128 B high every time (person_detect,
# ic_resnet, kws, ad). Activations are model-determined and identical in every
# build, so the whole gap is interpreter bookkeeping, and it is a fixed
# structural difference rather than anything that scales.
#
# That is 2.6% of person_detect's arena and 89% of the anomaly detector's, which
# is why this is stated in bytes. The old "~3%" caveat was measured on
# person_detect alone and was badly wrong for small models.
#
# Deliberately not subtracted. Against a Nano 33 BLE the same four deltas are
# 744, 1,304, 1,688 and 1,912, because Chirale's library is a different TFLM
# snapshot, so the constant belongs to a (TFLM version, target ABI) pair rather
# than to wasm32. Subtracting the ESP32 figure everywhere would under-report on
# other targets, and under-reporting is the unsafe direction for a fit check.
ESP32_OVERSHOOT_BYTES = 2128


def node_executable() -> str | None:
    return shutil.which("node")


def _caveat(total_bytes: int) -> str:
    """Say the overshoot in bytes, and what it is worth on this particular model."""
    share = 100 * ESP32_OVERSHOOT_BYTES / total_bytes if total_bytes else 0
    return (
        f"reads about {ESP32_OVERSHOOT_BYTES:,} B high against a real ESP32, a fixed "
        f"amount rather than a percentage, which is {share:.0f}% of this arena. "
        "Treat it as a ceiling."
    )


@dataclass(frozen=True)
class WasmArenaEstimator:
    """ArenaEstimator backed by the bundled wasm32 TFLM build."""

    node: str

    def estimate(self, model: ModelInfo) -> MemoryEstimate:
        with resources.as_file(resources.files("mcufit.wasm")) as wasm_dir:
            output = self._run(Path(wasm_dir) / SHIM, model.path)
        activations, overhead = parse_arena_table(output)
        return MemoryEstimate(
            peak_activation_bytes=activations,
            overhead_bytes=overhead,
            margin_bytes=0,
            peak_layer_index=-1,
            method="tflm-wasm32-measurement",
            caveat=_caveat(activations + overhead),
        )

    def _run(self, shim: Path, model_path: Path) -> str:
        try:
            proc = subprocess.run(
                [self.node, str(shim), str(model_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except OSError as exc:
            raise MeasurementUnavailableError(f"cannot run {self.node}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MeasurementUnavailableError("wasm TFLM benchmark timed out") from exc
        if proc.returncode != 0:
            raise MeasurementUnavailableError(
                "wasm TFLM benchmark failed on this model "
                f"(exit {proc.returncode}):\n{proc.stderr[-500:] or proc.stdout[-500:]}"
            )
        return proc.stdout + proc.stderr
