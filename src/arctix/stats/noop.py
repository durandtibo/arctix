from __future__ import annotations

__all__ = ["NoOpTracker"]

from typing import Any

from arctix.stats.base import BaseTracker


class NoOpTracker(BaseTracker[Any]):
    r"""Implements a no-operation statistics tracker."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def add(self, data: Any) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_statistics(self) -> dict:
        return {}
