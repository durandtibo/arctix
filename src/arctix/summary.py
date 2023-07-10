from __future__ import annotations

__all__ = ["BaseSummarizer", "Summarizer", "summary"]


from abc import ABC, abstractmethod
from typing import Any

from arctix.formatter import BaseFormatter
from arctix.utils.format import str_indent, str_mapping


def summary(
    value: Any,
    max_depth: int,
    max_items: int,
    num_spaces: int = 2,
    one_line: bool = False,
    summarizer: BaseSummarizer | None = None,
) -> str:
    return summarizer.summary(
        value=value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        num_spaces=num_spaces,
        one_line=one_line,
    )


class BaseSummarizer(ABC):
    @abstractmethod
    def summary(
        self,
        value: Any,
        depth: int,
        max_depth: int,
        max_items: int,
        num_spaces: int = 2,
        one_line: bool = False,
        summarizer: BaseSummarizer | None = None,
    ) -> str:
        pass


class Summarizer(BaseSummarizer):
    """Implements the default summarizer."""

    registry: dict[type[object], BaseFormatter] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_formatter(
        cls, data_type: type[object], formatter: BaseFormatter, exist_ok: bool = False
    ) -> None:
        r"""Adds a formatter for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            formatter (``BaseFormatter``): Specifies the formatter
                to use for the specified type.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the data type already exists. This
                parameter should be set to ``True`` to overwrite the
                formatter for a type. Default: ``False``.

        Raises:
            RuntimeError if a formatter is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from arctix import Summarizer, BaseFormatter, BaseSummarizer
            >>> class MyStringFormatter(BaseFormatter[str]):
            ...     def equal(
            ...         self,
            ...         tester: BaseEqualityTester,
            ...         object1: str,
            ...         object2: Any,
            ...         show_difference: bool = False,
            ...     ) -> bool:
            ...         return "<" + value + ">"  # Custom implementation to test strings
            ...
            >>> Summarizer.add_formatter(str, MyStringFormatter())
            # To overwrite an existing formatter
            >>> Summarizer.add_formatter(str, MyStringFormatter(), exist_ok=True)
        """
        if data_type in cls.registry and not exist_ok:
            raise RuntimeError(
                f"A formatter ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "formatter for this type"
            )
        cls.registry[data_type] = formatter

    def summary(
        self,
        value: Any,
        depth: int,
        max_depth: int,
        max_items: int,
        num_spaces: int = 2,
        one_line: bool = False,
        summarizer: BaseSummarizer | None = None,
    ) -> str:
        return self.find_formatter(type(value)).format(
            summarizer=self,
            value=value,
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            num_spaces=num_spaces,
            one_line=one_line,
        )

    @classmethod
    def has_formatter(cls, data_type: type[object]) -> bool:
        r"""Indicates if a formatter is registered for the given data
        type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if a formatter is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from arctix import Summarizer
            >>> Summarizer.has_formatter(list)
            True
            >>> Summarizer.has_formatter(str)
            False
        """
        return data_type in cls.registry

    @classmethod
    def find_formatter(cls, data_type: Any) -> BaseFormatter:
        r"""Finds the formatter associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseFormatter``: The formatter associated to the data type.

        Example usage:

        .. code-block:: pycon

            >>> from arctix import Summarizer
            >>> Summarizer.find_formatter(list)
            SequenceFormatter()
            >>> Summarizer.find_formatter(str)
            DefaultFormatter()
        """
        for object_type in data_type.__mro__:
            formatter = cls.registry.get(object_type, None)
            if formatter is not None:
                return formatter
        raise TypeError(f"Incorrect data type: {data_type}")
