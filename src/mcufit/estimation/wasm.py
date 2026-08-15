"""Arena measurement via TFLM compiled to wasm32, run under node.

wasm32 is 32-bit like the target, so interpreter overhead lands within ~3% of
a real device instead of the host build's ~8%. Needs node on PATH and nothing
else: the wasm is bundled in the wheel, so there is no `setup-exact` step.
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


def node_executable() -> str | None:
    return shutil.which("node")


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
            caveat=(
                "32-bit build, so overhead is close to the device's but not identical; "
                "measured ~3% high against a real ESP32"
            ),
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
