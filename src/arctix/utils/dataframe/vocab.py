r"""Contain utility functions for DataFrame and vocabulary."""

from __future__ import annotations

__all__ = ["generate_vocabulary"]

from collections import Counter
from typing import TYPE_CHECKING

from arctix.utils.vocab import Vocabulary

if TYPE_CHECKING:
    import polars as pl


def generate_vocabulary(frame: pl.DataFrame, col: str) -> Vocabulary:
    r"""Compute a vocabulary based on the content of a column.

    Args:
        frame: The input DataFrame.
        col: The name of the column used to compute the vocabulary.

    Returns:
        The generated vocabulary.

    Example usage:

    ```pycon

    >>> from collections import Counter
    >>> import polars as pl
    >>> from arctix.utils.dataframe import generate_vocabulary
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "a"],
    ...     }
    ... )
    >>> vocab1 = generate_vocabulary(frame, col="col1")
    >>> vocab1
    Vocabulary(
      counter=Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1}),
      index_to_token=(1, 2, 3, 4, 5),
      token_to_index={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    >>> vocab3 = generate_vocabulary(frame, col="col3")
    >>> vocab3
    Vocabulary(
      counter=Counter({'a': 2, 'b': 1, 'c': 1, 'd': 1}),
      index_to_token=('a', 'b', 'c', 'd'),
      token_to_index={'a': 0, 'b': 1, 'c': 2, 'd': 3},
    )

    ```
    """
    counts = frame.get_column(col).value_counts(sort=True)
    return Vocabulary(Counter(dict(zip(counts.get_column(col), counts.get_column("count")))))
