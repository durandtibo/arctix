r"""Contain transformers to replace values."""

from __future__ import annotations

__all__ = ["ReplaceDataFrameTransformer"]

from typing import Any

import polars as pl

from arctix.transformer.dataframe.base import BaseDataFrameTransformer


class ReplaceDataFrameTransformer(BaseDataFrameTransformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        orig_column: The original column name.
        final_column: The final column name.
        *args: The positional arguments to pass to ``replace``.
        **kwargs: The keyword arguments to pass to ``replace``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import Replace
    >>> transformer = Replace(
    ...     orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3}
    ... )
    >>> transformer
    ReplaceDataFrameTransformer(orig_column=old, final_column=new)
    >>> frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ old │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    │ d   │
    │ e   │
    └─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬─────┐
    │ old ┆ new │
    │ --- ┆ --- │
    │ str ┆ str │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    │ b   ┆ 2   │
    │ c   ┆ 3   │
    │ d   ┆ d   │
    │ e   ┆ e   │
    └─────┴─────┘
    >>> transformer = Replace(
    ...     orig_column="old",
    ...     final_column="new",
    ...     old={"a": 1, "b": 2, "c": 3},
    ...     default=None,
    ... )
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬──────┐
    │ old ┆ new  │
    │ --- ┆ ---  │
    │ str ┆ i64  │
    ╞═════╪══════╡
    │ a   ┆ 1    │
    │ b   ┆ 2    │
    │ c   ┆ 3    │
    │ d   ┆ null │
    │ e   ┆ null │
    └─────┴──────┘

    ```
    """

    def __init__(
        self,
        orig_column: str,
        final_column: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._orig_column = orig_column
        self._final_column = final_column
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(orig_column={self._orig_column}, "
            f"final_column={self._final_column})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        series = pl.col(self._orig_column)
        kwargs = self._kwargs.copy()
        if "default" not in kwargs:
            kwargs["default"] = series
        return frame.with_columns(
            series.replace_strict(*self._args, **kwargs).alias(self._final_column)
        )
