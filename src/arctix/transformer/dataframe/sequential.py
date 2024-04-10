r"""Contain a transformer to combine sequentially multiple
transformers."""

from __future__ import annotations

__all__ = ["SequentialDataFrameTransformer"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_sequence, str_indent, str_sequence

from arctix.transformer.dataframe.base import (
    BaseDataFrameTransformer,
    setup_dataframe_transformer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


class SequentialDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a ``polars.DataFrame`` transformer to apply
    sequentially several transformers.

    Args:
        transformers: The transformers or their configurations.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import (
    ...     Sequential,
    ...     Cast,
    ... )
    >>> transformer = Sequential(
    ...     [
    ...         Cast(columns=["col1"], dtype=pl.Float32),
    ...         Cast(columns=["col2"], dtype=pl.Int64),
    ...     ]
    ... )
    >>> transformer
    SequentialDataFrameTransformer(
      (0): CastDataFrameTransformer(columns=('col1',), dtype=Float32)
      (1): CastDataFrameTransformer(columns=('col2',), dtype=Int64)
    )
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
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ f32  ┆ i64  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1.0  ┆ 1    ┆ a     ┆ a     │
    │ 2.0  ┆ 2    ┆  b    ┆  b    │
    │ 3.0  ┆ 3    ┆   c   ┆   c   │
    │ 4.0  ┆ 4    ┆ d     ┆ d     │
    │ 5.0  ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘

    ```
    """

    def __init__(self, transformers: Sequence[BaseDataFrameTransformer | dict]) -> None:
        self._transformers = tuple(
            setup_dataframe_transformer(transformer) for transformer in transformers
        )

    def __repr__(self) -> str:
        args = ""
        if self._transformers:
            args = f"\n  {repr_indent(repr_sequence(self._transformers))}\n"
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = ""
        if self._transformers:
            args = f"\n  {str_indent(str_sequence(self._transformers))}\n"
        return f"{self.__class__.__qualname__}({args})"

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for transformer in self._transformers:
            frame = transformer.transform(frame)
        return frame
