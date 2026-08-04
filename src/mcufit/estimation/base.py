"""Estimator interface. Static analysis today; host-compiled TFLM
measurement will be an alternate implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.model import ModelInfo
from ..domain.report import MemoryEstimate


@runtime_checkable
class ArenaEstimator(Protocol):
    def estimate(self, model: ModelInfo) -> MemoryEstimate:
        """Estimate the tensor-arena RAM the model needs at runtime."""
        ...
