from __future__ import annotations

__all__ = ["str_mapping", "str_indent"]

from collections.abc import Mapping
from typing import Any


def str_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Computes a string representation of a mapping.

    Args:
    ----
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: pycon

        >>> from arctix.utils.format import str_mapping
        >>> str_mapping({"key1": "abc", "key2": "something\nelse"})
        (key1): abc
        (key2): something
          else
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}): {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: Specifies the original string. If the inputis not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: The indented string.

    Example usage:

    .. code-block:: pycon

        >>> from arctix.utils.format import str_indent
        >>> f"\t{str_indent('string1\nstring2\n  string3', 4)}"
            string1
            string2
              string3
    """
    formatted_str = str(original).split("\n")
    if len(formatted_str) == 1:  # single line
        return formatted_str[0]
    first = formatted_str.pop(0)
    formatted_str = "\n".join([(num_spaces * " ") + line for line in formatted_str])
    return first + "\n" + formatted_str
