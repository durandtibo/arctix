r"""Contain ``polars.DataFrame`` transformers to compute difference."""

from __future__ import annotations

__all__ = ["DiffDataFrameTransformer"]


import polars as pl

from arctix.transformer.dataframe import BaseDataFrameTransformer


class DiffDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to compute the first discrete difference
    between shifted items.

    Args:
        in_col: The input column name.
        out_col: The output column name.
        shift: The number of slots to shift.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import Diff
    >>> transformer = Diff(in_col="col1", out_col="diff")
    >>> transformer
    DiffDataFrameTransformer(in_col=col1, out_col=diff, shift=1)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ i64  ┆ str  │
    ╞══════╪══════╡
    │ 1    ┆ a    │
    │ 2    ┆ b    │
    │ 3    ┆ c    │
    │ 4    ┆ d    │
    │ 5    ┆ e    │
    └──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ diff │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ i64  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ a    ┆ null │
    │ 2    ┆ b    ┆ 1    │
    │ 3    ┆ c    ┆ 1    │
    │ 4    ┆ d    ┆ 1    │
    │ 5    ┆ e    ┆ 1    │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(self, in_col: str, out_col: str, shift: int = 1) -> None:
        self._in_col = in_col
        self._out_col = out_col
        self._shift = shift

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(in_col={self._in_col}, "
            f"out_col={self._out_col}, shift={self._shift})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            frame.select(pl.col(self._in_col).diff(n=self._shift).alias(self._out_col))
        )
