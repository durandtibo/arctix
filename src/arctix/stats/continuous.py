from __future__ import annotations

__all__ = ["ContinuousTracker"]

from collections import deque
from collections.abc import Sequence
from typing import Union

from coola.utils.imports import is_numpy_available, is_torch_available

from arctix.reduction import Reduction
from arctix.stats.base import BaseTracker, EmptyTrackerError

if is_numpy_available():
    import numpy as np
    from numpy import ndarray
else:  # pragma: no cover

    class ndarray:  # noqa: N801
        pass


if is_torch_available():
    from torch import Tensor
else:  # pragma: no cover

    class Tensor:
        pass


class ContinuousTracker(
    BaseTracker[Union[float, int, Sequence[Union[float, int]], Tensor, ndarray]]
):
    r"""Adds a statistics tracker for continuous values.

    This tracker computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values

    Args:
    ----
        max_size (int, optional): Specifies the maximum size used to
            store the last values because it may not be possible to
            store all the values. This parameter is used to compute
            the median and the quantiles. Default: ``10000``
        quantiles (`tuple or list, optional):
            Specifies a sequence of quantiles to compute, which must
            be between 0 and 1 inclusive. Default:
            ``(0.1, 0.25, 0.5, 0.75, 0.9)``
    """

    def __init__(
        self, max_size: int = 10000, quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
    ) -> None:
        self._sum = 0.0
        self._count = 0.0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._quantiles = tuple(sorted(quantiles))
        # Store only the N last values to scale to large number of values.
        self._values = deque(maxlen=max_size)
        self.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(max_size={self._values.maxlen:,}, "
            f"quantiles={self._quantiles})"
        )

    @property
    def values(self) -> tuple[float | int, ...]:
        return tuple(self._values)

    def add(self, data: float | int | Sequence[float | int] | Tensor | ndarray) -> None:
        if isinstance(data, Tensor):
            self._add_tensor(data)
        elif isinstance(data, ndarray):
            self._add_ndarray(data)
        elif isinstance(data, Sequence):
            self._add_sequence(data)
        else:
            self._add_scalar(data)

    def _add_ndarray(self, data: ndarray) -> None:
        if count := int(np.prod(data.shape)):
            values = data.flatten()
            self._sum += float(np.sum(values).item())
            self._count += count
            self._min_value = min(self._min_value, np.min(values).item())
            self._max_value = max(self._max_value, np.max(values).item())
            self._values.extend(values.tolist())

    def _add_scalar(self, data: float | int) -> None:
        value = data
        self._sum += value
        self._count += 1.0
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)
        self._values.append(value)

    def _add_sequence(self, data: Sequence[float | int]) -> None:
        if count := len(data):
            value = data
            self._sum += sum(value)
            self._count += count
            self._min_value = min(self._min_value, min(value))
            self._max_value = max(self._max_value, max(value))
            self._values.extend(value)

    def _add_tensor(self, data: Tensor) -> None:
        if count := data.numel():
            values = data.flatten()
            self._sum += float(values.sum().item())
            self._count += count
            self._min_value = min(self._min_value, values.min().item())
            self._max_value = max(self._max_value, values.max().item())
            self._values.extend(values.tolist())

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
        return int(self._count)

    def get_statistics(self) -> dict:
        if not self._count:
            raise EmptyTrackerError("Cannot compute the statistics because the tracker is empty")
        stats = {
            "count": self.count(),
            "sum": self.sum(),
            "mean": self.mean(),
            "median": self.median(),
            "std": self.std(),
            "max": self.max(),
            "min": self.min(),
        }
        stats.update(
            {
                f"quantile {quantile:.3f}": value
                for quantile, value in zip(self._quantiles, self.quantiles())
            }
        )
        return stats

    def max(self) -> float:
        r"""Gets the max value.

        Returns
        -------
            float: The max value.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.max()
            4
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the maximum because the tracker is empty")
        return self._max_value

    def mean(self) -> float:
        r"""Computes the mean value.

        This value is computed on all the values seen.

        Returns
        -------
            float: The mean value.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the mean because the tracker is empty")
        return self._sum / self._count

    def median(self) -> float:
        r"""Computes the median value from the last values.

        If there are more values than the maximum window size, only the
        last values are used. Internally, this tracker uses a deque to
        track the last values and the median value is computed on the
        values in the deque. The median is not unique for values with
        an even number of elements. In this case the lower of the two
        medians is returned.

        Returns
        -------
            float: The median value from the last values.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.median()
            2
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the median because the tracker is empty")
        return Reduction.reducer.median(self._values)

    def min(self) -> float:
        r"""Gets the min value.

        Returns
        -------
            float: The min value.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.min()
            1
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the minimum because the tracker is empty")
        return self._min_value

    def quantiles(self) -> list[float]:
        r"""Computes the quantiles.

        If there are more values than the maximum size, only the last
        values are used. Internally, this tracker uses a deque to
        track the last values and the quantiles are computed on the
        values in the deque.

        Returns
        -------
            list of floats: The standard deviation from the last
                values.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> tracker.quantiles()
            [1.0, 2.5, 5.0, 7.5, 9.0]
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the quantiles because the tracker is empty")
        return Reduction.reducer.quantiles(self._values, self._quantiles)

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0.0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._values.clear()

    def std(self) -> float:
        r"""Computes the standard deviation.

        If there are more values than the maximum size, only the last
        values are used. Internally, this tracker uses a deque to
        track the last values and the standard deviation is computed
        on the values in the deque.

        Returns
        -------
            float: The standard deviation from the last values.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.std()  # doctest: +ELLIPSIS
            1.527525...
        """
        if not self._count:
            raise EmptyTrackerError(
                "Cannot compute the standard deviation because the tracker is empty"
            )
        return Reduction.reducer.std(self._values)

    def sum(self) -> float:
        r"""Gets the sum value.

        Returns
        -------
            float: The sum value.

        Raises
        ------
            ``EmptyTrackerError`` if the statistics tracker is empty.

        Example usage:

        .. code-block:: pycon

            >>> from arctix.stats import ContinuousTracker
            >>> tracker = ContinuousTracker()
            >>> tracker.add([1, 2, 4])
            >>> tracker.sum()
            7.0
        """
        if not self._count:
            raise EmptyTrackerError("Cannot compute the sum because the tracker is empty")
        return self._sum
