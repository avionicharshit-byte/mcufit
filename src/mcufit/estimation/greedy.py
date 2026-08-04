"""Static estimate of peak tensor-arena usage.

A tensor is alive from the op that produces it to the last op that reads
it; the arena must hold all tensors alive at once, so the peak of that sum
over the schedule is what TFLM's planner has to pack. Per-op scratch
buffers (im2col etc.) are invisible to static analysis — covered by a
safety margin instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.model import ModelInfo
from ..domain.report import MemoryEstimate

_ARENA_ALIGNMENT = 16  # TFLM aligns tensor allocations to 16 bytes
_PER_TENSOR_OVERHEAD = 64  # approx. TfLiteTensor + allocation bookkeeping
_INTERPRETER_OVERHEAD = 4 * 1024  # interpreter structures living in the arena


def _align(size: int) -> int:
    return (size + _ARENA_ALIGNMENT - 1) // _ARENA_ALIGNMENT * _ARENA_ALIGNMENT


@dataclass(frozen=True)
class GreedyLifetimeEstimator:
    """ArenaEstimator implementation using static lifetime analysis."""

    scratch_margin: float = 0.20

    def estimate(self, model: ModelInfo) -> MemoryEstimate:
        lifetimes = self._tensor_lifetimes(model)

        peak_bytes = 0
        peak_layer = 0
        for step in range(max(1, len(model.layers))):
            live = sum(
                _align(size)
                for size, (start, end) in lifetimes.values()
                if start <= step <= end
            )
            if live > peak_bytes:
                peak_bytes = live
                peak_layer = step

        overhead = _INTERPRETER_OVERHEAD + _PER_TENSOR_OVERHEAD * len(model.tensors)
        margin = int(peak_bytes * self.scratch_margin)
        return MemoryEstimate(
            peak_activation_bytes=peak_bytes,
            overhead_bytes=overhead,
            margin_bytes=margin,
            peak_layer_index=peak_layer,
            method="static-lifetime-analysis",
        )

    @staticmethod
    def _tensor_lifetimes(model: ModelInfo) -> dict[int, tuple[int, tuple[int, int]]]:
        """Map activation tensor index -> (size, (first_step, last_step))."""
        activations = {t.index: t.size_bytes for t in model.activation_tensors}
        last_step = max(0, len(model.layers) - 1)

        lifetimes: dict[int, tuple[int, tuple[int, int]]] = {}
        for layer in model.layers:
            for idx in layer.output_tensors:
                if idx in activations and idx not in lifetimes:
                    lifetimes[idx] = (activations[idx], (layer.index, layer.index))
            for idx in layer.input_tensors:
                if idx in activations:
                    size, (start, _) = lifetimes.get(
                        idx, (activations[idx], (0, layer.index))
                    )
                    lifetimes[idx] = (size, (start, layer.index))

        # Graph inputs are written before the first op runs; graph outputs
        # must survive until after the last op finishes.
        for idx in model.graph_inputs:
            if idx in activations:
                size, (_, end) = lifetimes.get(idx, (activations[idx], (0, 0)))
                lifetimes[idx] = (size, (0, end))
        for idx in model.graph_outputs:
            if idx in activations:
                size, (start, _) = lifetimes.get(idx, (activations[idx], (0, last_step)))
                lifetimes[idx] = (size, (start, last_step))

        return lifetimes
