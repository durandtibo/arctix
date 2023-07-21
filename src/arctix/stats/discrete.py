from __future__ import annotations

__all__ = ["DiscreteTracker"]

from collections import Counter
from collections.abc import Sequence
from typing import Union

from coola.utils.imports import is_numpy_available, is_torch_available

from arctix.stats.base import BaseTracker, EmptyTrackerError

if is_numpy_available():
    from numpy import ndarray
else:  # pragma: no cover

    class ndarray:  # noqa: N801
        pass


if is_torch_available():
    from torch import Tensor
else:  # pragma: no cover

    class Tensor:
        pass


class DiscreteTracker(
    BaseTracker[
        Union[bool, float, int, str, Sequence[Union[bool, float, int, str]], Tensor, ndarray]
    ]
):
    r"""Implements a statistics tracker for discrete values.

    This statistics tracker computes the following statistics:

        - ``count``: the number of values
        - ``num_unique_values``: the number of unique values
        - ``count_{value}``: the number of values per unique value
    """

    def __init__(self) -> None:
        self._counter = Counter()
        self._total = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(total={self._total})"

    @property
    def counter(self) -> Counter:
        return self._counter

    def add(
        self, data: bool | float | int | str | Sequence[bool | float | int | str] | Tensor | ndarray
    ) -> None:
        if isinstance(data, str):  # need to be before Sequence
            self._add_scalar(data)
        elif isinstance(data, (Tensor, ndarray)):
            self._add_sequence(data.flatten().tolist())
        elif isinstance(data, Sequence):
            self._add_sequence(data)
        else:
            self._add_scalar(data)

    def _add_scalar(self, data: bool | float | int | str) -> None:
        self._counter[data] += 1
        self._total += 1

    def _add_sequence(self, data: Sequence[bool | float | int | str]) -> None:
        self._counter.update(data)
        self._total += len(data)

    def count(self) -> int:
        r"""Gets the number of values seen by the statistics tracker.

        Returns
        -------
            int: The number of values seen by the statistics tracker.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import DiscreteTracker
            >>> tracker = DiscreteTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.count()
            3
        """
        return self._total

    def get_statistics(self) -> dict:
        if not self.count():
            raise EmptyTrackerError("Cannot compute the statistics because the tracker is empty")
        summary = {
            "count": self.count(),
            "num_unique_values": len(self._counter),
        }
        for name, count in self.most_common():
            summary[f"count_{name}"] = count
        return summary

    def most_common(self, n: int | None = None) -> list[tuple[bool | float | int | str, int]]:
        r"""Gets a list of the ``n`` most common elements and their
        counts from the most common to the least.

        Args:
        ----
            n (int or None, optional): Specifies the number of
                elements to return. If ``n`` is ``None``, this
                method returns all elements in the counter.

        Returns:
        -------
            list: The list of the ``n`` most common elements and their
                counts. Elements with equal counts are ordered in the
                order first encountered.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import DiscreteTracker
            >>> tracker = DiscreteTracker()
            >>> tracker.add([1, 4, 1])
            >>> tracker.most_common()
            [(1, 2), (4, 1)]
        """
        if not self.count():
            raise EmptyTrackerError(
                "Cannot compute the most common values because the tracker is empty"
            )
        return self._counter.most_common(n)

    def reset(self) -> None:
        self._counter.clear()
        self._total = 0
