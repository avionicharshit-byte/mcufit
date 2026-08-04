"""Renderer interface — terminal and JSON today, HTML later."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.report import FitReport


@runtime_checkable
class ReportRenderer(Protocol):
    def render(self, report: FitReport) -> None:
        """Present a fit report to the user."""
        ...
