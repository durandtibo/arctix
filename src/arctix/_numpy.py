from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from arctix.formatter import BaseFormatter
from arctix.summarizer import BaseSummarizer, Summarizer
from arctix.utils.imports import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy
else:
    numpy = Mock()  # pragma: no cover


class NDArrayFormatter(BaseFormatter[numpy.ndarray]):
    r"""Implement a formatter for ``numpy.ndarray``.

    Args:
    ----
        show_data (bool, optional): If ``True``, the returned string
            is the default string representation. `If ``False``
            the returned string only contains the tensor metadata.
            Default: ``False``
    """

    def __init__(self, show_data: bool = False) -> None:
        check_numpy()
        self._show_data = bool(show_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(show_data={self._show_data})"

    def clone(self) -> NDArrayFormatter:
        return self.__class__(show_data=self._show_data)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._show_data == other._show_data

    def format(
        self, summarizer: BaseSummarizer, value: numpy.ndarray, depth: int = 0, max_depth: int = 1
    ) -> str:
        if self._show_data:
            return repr(value)
        return " | ".join(
            [
                f"{type(value)}",
                f"shape={value.shape}",
                f"dtype={value.dtype}",
            ]
        )

    def load_state_dict(self, state: dict) -> None:
        self._show_data = state["show_data"]

    def state_dict(self) -> dict:
        return {"show_data": self._show_data}

    def get_show_data(self) -> bool:
        r"""Indicates if the tensor data or metadata are shown.

        Returns:
        -------
            bool: ``True`` if the tensor data are shown,
                ``False`` if the tensor metadata are shown.

        Example usage:

        .. code-block:: pycon

            >>> from arctix._numpy import NDArrayFormatter
            >>> formatter = NDArrayFormatter()
            >>> formatter.get_show_data()
            False
        """
        return self._show_data

    def set_show_data(self, show_data: bool) -> None:
        r"""Set if the tensor data or metadata are shown.

        Args:
        ----
            show_data (bool): ``True`` if the tensor data are shown,
                ``False`` if the tensor metadata are shown.

        Raises:
        ------
            TypeError if ``show_data`` is not an boolean.

        Example usage:

        .. code-block:: pycon

            >>> from arctix._numpy import NDArrayFormatter
            >>> formatter = NDArrayFormatter()
            >>> formatter.set_show_data(True)
            >>> formatter.get_show_data()
            True
        """
        if not isinstance(show_data, bool):
            raise TypeError(
                "Incorrect type for show_data. Expected bool value but received {show_data}"
            )
        self._show_data = show_data


if is_numpy_available():  # pragma: no cover
    if not Summarizer.has_formatter(numpy.ndarray):
        Summarizer.add_formatter(numpy.ndarray, NDArrayFormatter())
