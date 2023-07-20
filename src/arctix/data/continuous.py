r"""This module implements a data summary for continuous numerical
values."""

from __future__ import annotations

__all__ = [
    "BaseContinuousDataSummary",
    "FloatDataSummary",
    "FloatTensorDataSummary",
    "FloatTensorSequenceDataSummary",
]

from collections import deque
from typing import TypeVar
from unittest.mock import Mock

from arctix.data.base import BaseDataSummary, EmptyDataSummaryError
from arctix.data.sequence import BaseSequenceDataSummary
from arctix.reduction import Reduction
from arctix.utils.imports import check_torch, is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover

T = TypeVar("T")

DEFAULT_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class BaseContinuousDataSummary(BaseDataSummary[T]):
    r"""Implements a data summary for continuous numerical values.

    This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values

    A child class has to implement the ``add`` method.

    Args:
    ----
        max_size (int, optional): Specifies the maximum size used to
            store the last values because it may not be possible to
            store all the values. This parameter is used to compute
            the median and the quantiles. Default: ``10000``
        quantiles (`tuple or list, optional):
            Specifies a sequence of quantiles to compute, which must
            be between 0 and 1 inclusive. Default:
            ``(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)``
    """

    def __init__(
        self,
        max_size: int = 10000,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    ) -> None:
        check_torch()
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

    def count(self) -> int:
        r"""Gets the number of values seen by the summary.

        Returns
        -------
            int: The number of values seen by the summary.
        """
        return int(self._count)

    def max(self) -> float:
        r"""Gets the max value.

        Returns
        -------
            float: The max value.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._max_value

    def mean(self) -> float:
        r"""Computes the mean value.

        This value is computed on all the values seen.

        Returns
        -------
            float: The mean value.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._sum / self._count

    def median(self) -> float:
        r"""Computes the median value from the last values.

        If there are more values than the maximum window size, only the
        last values are used. Internally, this summary uses a deque to
        track the last values and the median value is computed on the
        values in the deque. The median is not unique for values with
        an even number of elements. In this case the lower of the two
        medians is returned.

        Returns
        -------
            float: The median value from the last values.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return Reduction.reducer.median(self._values)

    def min(self) -> float:
        r"""Gets the min value.

        Returns
        -------
            float: The min value.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._min_value

    def quantiles(self) -> list[int | float]:
        r"""Computes the quantiles.

        If there are more values than the maximum size, only the last
        values are used. Internally, this summary uses a deque to
        track the last values and the quantiles are computed on the
        values in the deque.

        Returns
        -------
            float: The standard deviation from the last values.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return Reduction.reducer.quantiles(self._values, self._quantiles)

    def reset(self) -> None:
        r"""Resets the data summary."""
        self._sum = 0.0
        self._count = 0.0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._values.clear()

    def std(self) -> float:
        r"""Computes the standard deviation.

        If there are more values than the maximum size, only the last
        values are used. Internally, this summary uses a deque to
        track the last values and the standard deviation is computed
        on the values in the deque.

        Returns
        -------
            float: The standard deviation from the last values.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return Reduction.reducer.std(self._values)

    def sum(self) -> float:
        r"""Gets the sum value.

        Returns
        -------
            float: The sum value.

        Raises
        ------
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._sum

    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns
        -------
            dict: The data descriptive summary.

        Raises
        ------
            ``EmptyDataSummaryError`` is the data summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        summary = {
            "count": self.count(),
            "sum": self.sum(),
            "mean": self.mean(),
            "median": self.median(),
            "std": self.std(),
            "max": self.max(),
            "min": self.min(),
        }
        summary.update(
            {
                f"quantile {quantile:.3f}": value
                for quantile, value in zip(self._quantiles, self.quantiles())
            }
        )
        return summary


class FloatDataSummary(BaseContinuousDataSummary[float]):
    r"""Implements a data summary for float values.

    This data summary assumes that the data are continuous numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of the values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values
    """

    def add(self, data: float) -> None:
        r"""Adds new data to the summary.

        Args:
        ----
            data (float): Specifies the data to add to the summary.
        """
        value = float(data)
        self._sum += value
        self._count += 1.0
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)
        self._values.append(value)


class FloatTensorDataSummary(BaseContinuousDataSummary[torch.Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type float.

    This data summary assumes that the data are continuous numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values
    """

    def add(self, data: torch.Tensor) -> None:
        r"""Adds new data to the summary.

        Args:
        ----
            data (``torch.Tensor`` of type float): Specifies the data
                to add to the summary. This method converts the input
                to a ``torch.Tensor`` of type float if the tensor type
                is different.
        """
        if data.numel() > 0:
            values = data.float().flatten()
            self._sum += float(values.sum())
            self._count += values.numel()
            self._min_value = min(self._min_value, values.min().item())
            self._max_value = max(self._max_value, values.max().item())
            self._values.extend(values.tolist())


class FloatTensorSequenceDataSummary(BaseSequenceDataSummary[torch.Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type float.

    The input should have a shape ``(sequence_length, *)`` where `*`
    means any number of dimensions. This data summary assumes that
    the data are continuous numerical values. This data summary
    computes the following statistics:

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
        quantiles (tuple or list, optional):
            Specifies a sequence of quantiles to compute, which must
            be between 0 and 1 inclusive. Default:
            ``(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)``
    """

    def __init__(
        self,
        max_size: int = 10000,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__(value=FloatTensorDataSummary(max_size=max_size, quantiles=quantiles))

    def _get_sequence_length(self, data: torch.Tensor) -> int:
        r"""Gets the sequence length of the data.

        Args:
        ----
            data (``torch.Tensor`` of type float and shape
                ``(sequence_length, *)`` where `*` means any number of
                dimensions): Specifies the input sequence.

        Returns:
        -------
            int: The sequence length.
        """
        return data.shape[0]  # TODO: make seq_dim configurable
