"""Parser abstraction.

New model formats (ONNX, CoreML, ...) are added by implementing this
protocol — nothing downstream changes (open/closed principle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..domain.model import ModelInfo


class ModelParseError(Exception):
    """Raised when a model file cannot be understood."""


@runtime_checkable
class ModelParser(Protocol):
    def supports(self, path: Path) -> bool:
        """Whether this parser can handle the given file."""
        ...

    def parse(self, path: Path) -> ModelInfo:
        """Parse the model file into a ModelInfo. Raises ModelParseError."""
        ...
