"""Core domain objects describing a parsed ML model.

These are pure data structures: no parsing, no I/O, no estimation logic.
Every other layer of mcufit depends on these, never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Quantization(str, Enum):
    FLOAT32 = "float32"
    INT8 = "int8"
    FLOAT16 = "float16"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorInfo:
    """A single tensor in the model graph."""

    index: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    is_weight: bool
    """True if the tensor has static data baked into the model (lives in
    flash), False if it is an activation computed at runtime (lives in RAM)."""


@dataclass(frozen=True)
class LayerInfo:
    """A single operator in the model's execution order."""

    index: int
    op_name: str
    input_tensors: tuple[int, ...]
    output_tensors: tuple[int, ...]


@dataclass(frozen=True)
class ModelInfo:
    """Everything mcufit knows about a model after parsing."""

    path: Path
    file_size_bytes: int
    quantization: Quantization
    tensors: tuple[TensorInfo, ...] = field(repr=False)
    layers: tuple[LayerInfo, ...] = field(repr=False)
    graph_inputs: tuple[int, ...] = ()
    graph_outputs: tuple[int, ...] = ()

    @property
    def weights_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors if t.is_weight)

    @property
    def activation_tensors(self) -> tuple[TensorInfo, ...]:
        return tuple(t for t in self.tensors if not t.is_weight)

    @property
    def op_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for layer in self.layers:
            counts[layer.op_name] = counts.get(layer.op_name, 0) + 1
        return counts
