from __future__ import annotations

__all__ = ["BaseFormatter", "DefaultFormatter"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from arctix.summarizer import BaseSummarizer

T = TypeVar("T")


class BaseFormatter(ABC, Generic[T]):
    @abstractmethod
    def clone(self) -> BaseFormatter:
        r"""Returns a copy of the formatter.

        Returns:
            ``BaseFormatter``: A copy of the formatter.
        """

    @abstractmethod
    def format(self, summarizer: BaseSummarizer, value: T, depth: int, max_depth: int) -> str:
        r"""Format a value.

        Args:
        ----
            summarizer (``BaseSummarizer``): Specifies the summarizer.
            value: Specifies the value to summarize.

        Returns:
        -------
            str: The formatted value.
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example:
        -------
        .. code-block:: pycon

            >>> from arctix.formatter import DefaultFormatter
            >>> formatter = DefaultFormatter()
            >>> # Please take a look to the implementation of the state_dict
            >>> # function to know the expected structure
            >>> formatter.load_state_dict({"max_characters": 10})
            >>> formatter
            DefaultFormatter(max_characters=10)
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
        -------
            dict: the state values in a dict.

        Example:
        -------
        .. code-block:: pycon

            >>> from arctix.formatter import DefaultFormatter
            >>> formatter = DefaultFormatter()
            >>> formatter.state_dict()
            {'max_characters': -1}
        """


class DefaultFormatter(BaseFormatter[Any]):
    r"""Implements the default formatter.

    Args:
        max_characters (int, optional): Specifies the maximum number
            of characters to show. If a negative value is provided,
            all the characters are shown. Default: ``-1``
    """

    def __init__(self, max_characters: int = -1) -> None:
        self.set_max_characters(max_characters)

    def __repr__(self) -> str:
        max_characters = (
            self._max_characters if self._max_characters is None else f"{self._max_characters:,}"
        )
        return f"{self.__class__.__qualname__}(max_characters={max_characters})"

    def clone(self) -> DefaultFormatter:
        return self.__class__(max_characters=self._max_characters)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._max_characters == other._max_characters

    def format(
        self, summarizer: BaseSummarizer, value: Any, depth: int = 0, max_depth: int = 1
    ) -> str:
        typ = type(value)
        if self._max_characters >= 0:
            value = str(value)
            if len(value) > self._max_characters:
                value = value[: self._max_characters] + "..."
        return f"{typ} {value}"

    def load_state_dict(self, state: dict) -> None:
        self._max_characters = state["max_characters"]

    def state_dict(self) -> dict:
        return {"max_characters": self._max_characters}

    def get_max_characters(self) -> int:
        return self._max_characters

    def set_max_characters(self, max_characters: int) -> None:
        if not isinstance(max_characters, int):
            raise TypeError(
                "Incorrect type for max_characters. Expected int value but "
                f"received {max_characters}"
            )
        self._max_characters = max_characters
