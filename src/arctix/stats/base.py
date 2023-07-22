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

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add(1)
            >>> tracker.add(4.2)
            >>> tracker.mean()
            2.6
        """

    @abstractmethod
    def count(self) -> int:
        r"""Gets the number of values seen by the statistics tracker.

        Returns
        -------
            int: The number of values seen by the statistics tracker.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.count()
            3
        """

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

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 4.2])
            >>> tracker.get_statistics()  # doctest: +ELLIPSIS
            {'count': 2, 'sum': 5.2, 'mean': 2.6, ...}
        """

    @abstractmethod
    def reset(self) -> None:
        r"""Resets the statistics tracker.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 4.2])
            >>> tracker.reset()
            >>> tracker.count()
            0
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Load the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> # Please take a look to the implementation of the state_dict
            >>> # function to know the expected structure
            >>> tracker.load_state_dict(
            ...     {
            ...         "count": 3,
            ...         "max_value": 4,
            ...         "min_value": 1,
            ...         "quantiles": (0.1, 0.25, 0.5, 0.75, 0.9),
            ...         "sum": 7.0,
            ...         "values": (1, 2, 4),
            ...     }
            ... )
            >>> tracker.state_dict()  # doctest: +ELLIPSIS
            {'count': 3, 'max_value': 4, 'min_value': 1, ...}
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Return a dictionary containing state values.

        Example usage:
            dict: the state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.state_dict()  # doctest: +ELLIPSIS
            {'count': 0, 'max_value': -inf, 'min_value': inf, ...}
        """


class EmptyTrackerError(Exception):
    r"""Raised when the tracker is empty because it is not possible to
    compute stats without seeing data."""
