r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = ["convert_to_dict_of_flat_lists", "sort_by_key", "sort_by_value"]

from typing import TYPE_CHECKING

from coola.nested import convert_to_dict_of_lists

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping


def convert_to_dict_of_flat_lists(
    seq_of_mappings: list[dict[Hashable, list]],
) -> dict[Hashable, list]:
    r"""Convert a sequence of mappings to a dictionary of lists.

    All the dictionaries should have the same keys. The first
    mapping in the sequence is used to find the keys.
    The lists of lists are converted to flat lists.

    Args:
        seq_of_mappings: The sequence of mappings to convert.

    Returns:
        A dictionary of lists.

    Example usage:

    ```pycon

    >>> from arctix.utils.mapping import convert_to_dict_of_flat_lists
    >>> convert_to_dict_of_flat_lists(
    ...     [
    ...         {"key1": [1, 2], "key2": [10, 11]},
    ...         {"key1": [2], "key2": [20]},
    ...         {"key1": [3, 4, 5], "key2": [30, 31, 32]},
    ...     ]
    ... )
    {'key1': [1, 2, 2, 3, 4, 5], 'key2': [10, 11, 20, 30, 31, 32]}

    ```
    """
    mapping = convert_to_dict_of_lists(seq_of_mappings)
    return {key: [v for value in values for v in value] for key, values in mapping.items()}


def sort_by_key(data: Mapping, reverse: bool = False) -> dict:
    r"""Sort the mapping by keys.

    Args:
        data: The mapping to sort.
        reverse: If ``True``, the keys are sorted in reverse order.

    Returns:
        A data where the keys are sorted.

    Example usage:

    ```pycon

    >>> from arctix.utils.mapping import sort_by_key
    >>> sort_by_key({"b": 3, "c": 1, "a": 2})
    {'a': 2, 'b': 3, 'c': 1}
    >>> sort_by_key({"b": 3, "c": 1, "a": 2}, reverse=True)
    {'c': 1, 'b': 3, 'a': 2}

    ```
    """
    return dict(sorted(data.items(), reverse=reverse))


def sort_by_value(data: Mapping, reverse: bool = False) -> dict:
    r"""Sort the mapping by values.

    Args:
        data: The mapping to sort.
        reverse: If ``True``, the values are sorted in reverse order.

    Returns:
        A data where the values are sorted.

    Example usage:

    ```pycon

    >>> from arctix.utils.mapping import sort_by_value
    >>> sort_by_value({"b": 3, "c": 1, "a": 2, "d": 2})
    {'c': 1, 'a': 2, 'd': 2, 'b': 3}
    >>> sort_by_value({"b": 3, "c": 1, "a": 2, "d": 2}, reverse=True)
    {'b': 3, 'a': 2, 'd': 2, 'c': 1}

    ```
    """
    return dict(sorted(data.items(), key=lambda kv: kv[1], reverse=reverse))
