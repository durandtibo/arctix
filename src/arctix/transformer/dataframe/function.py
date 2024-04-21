r"""Contain a transformer that is a wrapper around a function to
transform the DataFrame."""

from __future__ import annotations

__all__ = ["FunctionDataFrameTransformer"]

from typing import TYPE_CHECKING

from arctix.transformer.dataframe.base import BaseDataFrameTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl


class FunctionDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer that is a wrapper around a function to
    transform the DataFrame.

    Args:
        func: The function to transform the DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import FunctionDataFrameTransformer
    >>> transformer = FunctionDataFrameTransformer(
    ...     func=lambda frame: frame.filter(pl.col("col1").is_in({2, 4}))
    ... )
    >>> transformer
    FunctionDataFrameTransformer(func=<function <lambda> at 0x...>)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> out = transformer.transform(frame)
    >>> out
    shape: (2, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(self, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
        self._func = func

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(func={self._func})"

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return self._func(frame)
