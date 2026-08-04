"""Renderer abstraction — terminal, JSON, and future formats (HTML,
GitHub-comment markdown) all implement this."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.report import FitReport


@runtime_checkable
class ReportRenderer(Protocol):
    def render(self, report: FitReport) -> None:
        """Present a fit report to the user."""
        ...
