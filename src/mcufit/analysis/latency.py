"""Order-of-magnitude inference latency from MAC counts.

Counts multiply-accumulates for the compute-heavy ops and divides by the
board's rough throughput (clock x MACs/cycle for its core family). Real
latency depends on kernels, memory placement, and quantization — this is
for "milliseconds or seconds?" answers, and is labelled accordingly.
"""

from __future__ import annotations

from ..domain.board import Board
from ..domain.model import LayerInfo, ModelInfo, TensorInfo


def count_macs(model: ModelInfo) -> int:
    tensors = {t.index: t for t in model.tensors}
    return sum(_layer_macs(layer, tensors) for layer in model.layers)


def estimate_latency_ms(model: ModelInfo, board: Board) -> float | None:
    if board.cpu_mhz <= 0 or board.macs_per_cycle <= 0:
        return None
    macs = count_macs(model)
    if macs == 0:
        return None
    return macs / (board.cpu_mhz * 1e6 * board.macs_per_cycle) * 1000


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
