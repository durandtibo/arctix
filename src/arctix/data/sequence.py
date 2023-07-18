r"""This module implements a base class to compute a descriptive summary
of sequences."""
from __future__ import annotations

__all__ = ["BaseSequenceDataSummary"]

from abc import abstractmethod
from typing import TypeVar

from arctix.data.base import BaseDataSummary

T = TypeVar("T")


class BaseSequenceDataSummary(BaseDataSummary[T]):
    r"""Implements a base class to compute a descriptive summary of
    sequences.

    A child class has to implement the ``_get_sequence_length`` method.

    Args:
    ----
        value (``BaseDataSummary``): Specifies the summary
            object used to compute a descriptive summary of the
            values in the sequences.
    """

    def __init__(self, value: BaseDataSummary[T]) -> None:
        self._value = value

        # local import to avoid cyclic dependency
        from arctix.data import FloatDataSummary

        self._length = FloatDataSummary()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  value={self._value},\n"
            f"  length={self._length}\n"
            ")"
        )

    def add(self, data: T) -> None:
        r"""Adds new data to the summary.

        Args:
        ----
            data: Specifies the data to add to the summary.
        """
        self._value.add(data)
        self._length.add(self._get_sequence_length(data))

    def reset(self) -> None:
        r"""Resets the data summary."""
        self._value.reset()
        self._length.reset()

    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns
        -------
            dict: The data descriptive summary. The dictionary has two
                keys: ``'length'`` and ``'value'``. The key
                ``'length'`` contains some information about the
                sequence length. The key ``'value'`` contains some
                information about the values in the sequence.

        Raises
        ------
            ``EmptyDataSummaryError`` is the data summary is empty.
        """
        return {
            "value": self._value.summary(),
            "length": self._length.summary(),
        }

    @abstractmethod
    def _get_sequence_length(self, data: T) -> int:
        r"""Gets the sequence length of the data.

        Args:
        ----
            data : Specifies the input sequence.

        Returns:
        -------
            int: The sequence length.
        """
