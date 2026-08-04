"""Estimator abstraction.

The static analyzer is the default implementation. A future measurement
mode (running the real TFLM interpreter compiled for the host) plugs in
here without touching any other layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.model import ModelInfo
from ..domain.report import MemoryEstimate


@runtime_checkable
class ArenaEstimator(Protocol):
    def estimate(self, model: ModelInfo) -> MemoryEstimate:
        """Estimate the tensor-arena RAM the model needs at runtime."""
        ...
