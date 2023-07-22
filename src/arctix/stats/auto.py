from __future__ import annotations

__all__ = ["AutoTracker"]

from collections.abc import Sequence
from typing import Any

from coola.utils import is_torch_available, str_indent

from arctix.stats.base import BaseTracker, EmptyTrackerError
from arctix.stats.continuous import ContinuousTracker
from arctix.stats.discrete import DiscreteTracker
from arctix.utils.types import Tensor, ndarray

if is_torch_available():  # pragma: no cover
    import torch


class AutoTracker(BaseTracker[Any]):
    r"""Implements a statistics tracker that tracks stats based on the
    data seen  when ``add`` is called the first time.

    The internal tracker is initialized based on the data when
    ``add`` is called the first time.
    """

    def __init__(self) -> None:
        self._tracker: BaseTracker | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  tracker={str_indent(self._tracker)}\n)"

    @property
    def tracker(self) -> BaseTracker | None:
        r"""``BaseTracker``: The internal tracker."""
        return self._tracker

    def add(self, data: Any) -> None:
        if self._tracker is None:
            self._tracker = self._initialize_tracker_from_data(data)
        self._tracker.add(data)

    def count(self) -> int:
        if self._tracker is None:
            return 0
        return self._tracker.count()

    def get_statistics(self) -> dict:
        if self._tracker is None:
            raise EmptyTrackerError("Cannot compute the statistics because the tracker is empty")
        return self._tracker.get_statistics()

    def reset(self) -> None:
        if self._tracker:
            self._tracker.reset()

    def load_state_dict(self, state_dict: dict) -> None:
        if self._tracker:
            self._tracker.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        if self._tracker is None:
            return {}
        return self._tracker.state_dict()

    def _initialize_tracker_from_data(self, data: Any) -> BaseTracker:
        if isinstance(data, Tensor):
            if data.dtype in (torch.float, torch.double, torch.int, torch.long):
                return ContinuousTracker()
            return DiscreteTracker()
        if isinstance(data, ndarray):
            if data.dtype in (float, int):
                return ContinuousTracker()
            return DiscreteTracker()
        if isinstance(data, (bool, str)):
            return DiscreteTracker()
        if isinstance(data, (float, int)):
            return ContinuousTracker()
        if isinstance(data, Sequence) and all(isinstance(value, (float, int)) for value in data):
            return ContinuousTracker()
        return DiscreteTracker()
