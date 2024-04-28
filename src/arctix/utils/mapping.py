r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = ["convert_to_dict_of_flat_lists"]


from typing import TYPE_CHECKING

from coola.nested import convert_to_dict_of_lists

if TYPE_CHECKING:
    from collections.abc import Hashable


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
