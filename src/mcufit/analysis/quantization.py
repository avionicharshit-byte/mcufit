"""Simulates int8 quantization by transforming the parsed model.

Instead of guessing "divide by 4", shrink every float tensor to one byte
per element - what int8 conversion actually does - and re-run the same
lifetime analysis on the transformed graph. Biases stay int32, exactly as
real converters keep them.
"""

from __future__ import annotations

from dataclasses import replace

from ..domain.model import ModelInfo, Quantization, TensorInfo


def project_int8(model: ModelInfo) -> ModelInfo:
    tensors = tuple(_shrink(t) for t in model.tensors)
    return replace(model, tensors=tensors, quantization=Quantization.INT8)


def projected_file_size(model: ModelInfo) -> int:
    """Model file size after quantization: float weights drop to 1/4."""
    float_weight_bytes = sum(
        t.size_bytes for t in model.tensors if t.is_weight and t.dtype in ("float32", "float64")
    )
    return model.file_size_bytes - (float_weight_bytes * 3) // 4


def _shrink(tensor: TensorInfo) -> TensorInfo:
    if tensor.dtype not in ("float32", "float64"):
        return tensor
    elements = 1
    for dim in tensor.shape:
        elements *= dim
    return replace(tensor, dtype="int8", size_bytes=elements)
