from __future__ import annotations

__all__ = ["BaseTracker", "EmptyTrackerError"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class BaseTracker(Generic[T], ABC):
    r"""Defines the base class to implement a statistics tracker.

    Note that the statistics tracker only stores some data statistics,
    and not the data.
    """

    @abstractmethod
    def add(self, data: T) -> None:
        r"""Adds new data to the statistics tracker.

        Args:
        ----
            data: Specifies the data to add to the statistics tracker.
        """

    @abstractmethod
    def reset(self) -> None:
        r"""Resets the statistics tracker."""

    @abstractmethod
    def get_statistics(self) -> dict:
        r"""Gets the statistics.

        Note that the statistics depends on the data type.

        Returns
        -------
            dict: The statistics.

        Raises
        ------
            ``EmptyTrackerError`` if the tracker is empty.
        """


class EmptyTrackerError(Exception):
    r"""Raised when the tracker is empty because it is not possible to
    compute stats without seeing data."""
