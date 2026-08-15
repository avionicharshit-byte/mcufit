"""Inference latency from per-operator throughput measured on real hardware.

Only boards in `boards/data/measured.yaml` get an answer. Everything else
returns None, because speed cannot be derived from a datasheet: which operators
a chip runs fast depends on which kernels its vendor happened to write, and
that differs per chip. On the two boards measured so far, fully-connected is
3.2x slower than convolution on ESP32 while depthwise is 3.3x slower than
convolution on the nRF52840.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml

from ..domain.board import Board
from ..domain.model import LayerInfo, ModelInfo, TensorInfo


@dataclass(frozen=True)
class LatencyEstimate:
    milliseconds: float
    board_id: str
    measured_on: str
    kernels: str
    error_pct: tuple[float, float]
    unmodelled_ops: tuple[str, ...]
    """Operators in the model with no measurement, so excluded from the total."""

    @property
    def accuracy_note(self) -> str:
        low, high = self.error_pct
        return f"measured {self.measured_on} on {self.kernels}; {low:+.0f}% to {high:+.0f}%"


@lru_cache(maxsize=1)
def _measurements() -> dict:
    data = resources.files("mcufit.boards.data").joinpath("measured.yaml")
    return yaml.safe_load(data.read_text()).get("measurements", {})


def measured_boards() -> list[str]:
    return sorted(_measurements())


def count_macs(model: ModelInfo) -> int:
    return sum(macs_by_op(model).values())


def macs_by_op(model: ModelInfo) -> dict[str, int]:
    tensors = {t.index: t for t in model.tensors}
    out: dict[str, int] = defaultdict(int)
    for layer in model.layers:
        out[layer.op_name] += _layer_macs(layer, tensors)
    return dict(out)


def calls_by_op(model: ModelInfo) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for layer in model.layers:
        out[layer.op_name] += 1
    return dict(out)


def estimate_latency(model: ModelInfo, board: Board) -> LatencyEstimate | None:
    entry = _measurements().get(board.id)
    if entry is None:
        return None

    per_cycle = entry.get("macs_per_cycle", {})
    per_call = entry.get("us_per_call", {})
    mhz = entry.get("cpu_mhz") or board.cpu_mhz
    if not mhz:
        return None

    macs, calls = macs_by_op(model), calls_by_op(model)
    microseconds = 0.0
    unmodelled = []
    for op in calls:
        if op in per_cycle and per_cycle[op] > 0:
            microseconds += macs.get(op, 0) / per_cycle[op] / mhz
        elif op in per_call:
            microseconds += per_call[op] * calls[op]
        else:
            unmodelled.append(op)

    low, high = entry.get("error_pct", [0, 0])
    return LatencyEstimate(
        milliseconds=microseconds / 1000,
        board_id=board.id,
        measured_on=str(entry.get("measured_on", "unknown")),
        kernels=str(entry.get("kernels", "unknown")),
        error_pct=(float(low), float(high)),
        unmodelled_ops=tuple(sorted(unmodelled)),
    )


def _layer_macs(layer: LayerInfo, tensors: dict[int, TensorInfo]) -> int:
    out = tensors.get(layer.output_tensors[0]) if layer.output_tensors else None
    if out is None:
        return 0
    out_elems = 1
    for dim in out.shape:
        out_elems *= dim

    weight = next(
        (tensors[i] for i in layer.input_tensors if i in tensors and tensors[i].is_weight
         and len(tensors[i].shape) >= 2),
        None,
    )

    if layer.op_name == "CONV_2D" and weight is not None and len(weight.shape) == 4:
        _, kh, kw, in_c = weight.shape
        return out_elems * kh * kw * in_c
    if layer.op_name == "DEPTHWISE_CONV_2D" and weight is not None and len(weight.shape) == 4:
        _, kh, kw, _ = weight.shape
        return out_elems * kh * kw
    if layer.op_name == "FULLY_CONNECTED" and weight is not None:
        rows, cols = weight.shape[0], weight.shape[-1]
        return rows * cols
    # Element-wise and shape ops: roughly one op per output element.
    return out_elems
