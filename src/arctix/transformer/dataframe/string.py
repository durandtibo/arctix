r"""Contain ``polars.DataFrame`` transformers to process string
values."""

from __future__ import annotations

__all__ = ["StripCharsDataFrameTransformer"]

from typing import TYPE_CHECKING

import polars as pl

from arctix.transformer.dataframe.base import BaseDataFrameTransformer
from arctix.utils.imports import is_tqdm_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from arctix.utils.noop import tqdm


class StripCharsDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to remove leading and trailing
    characters.

    Args:
        columns: The columns to prepare.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import StripChars
    >>> transformer = StripChars(columns=["col2", "col3"])
    >>> transformer
    StripCharsDataFrameTransformer(columns=('col2', 'col3'))
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ str  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1    ┆ 1    ┆ a     ┆ a     │
    │ 2    ┆ 2    ┆  b    ┆  b    │
    │ 3    ┆ 3    ┆   c   ┆   c   │
    │ 4    ┆ 4    ┆ d     ┆ d     │
    │ 5    ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ str  ┆ str  ┆ str   │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 1    ┆ a    ┆ a     │
    │ 2    ┆ 2    ┆ b    ┆  b    │
    │ 3    ┆ 3    ┆ c    ┆   c   │
    │ 4    ┆ 4    ┆ d    ┆ d     │
    │ 5    ┆ 5    ┆ e    ┆ e     │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self._columns = tuple(columns)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(columns={self._columns})"

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in tqdm(self._columns, desc="stripping chars"):
            frame = frame.with_columns(frame.select(pl.col(col).str.strip_chars()))
        return frame
