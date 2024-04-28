r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["JsonDecodeDataFrameTransformer"]

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


class JsonDecodeDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to parse string values as JSON.

    Args:
        columns: The columns to parse.
        dtype: The dtype to cast the extracted value to.
            If ``None``, the dtype will be inferred from the JSON value.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.transformer.dataframe import JsonDecode
    >>> transformer = JsonDecode(columns=["col1", "col3"])
    >>> transformer
    JsonDecodeDataFrameTransformer(columns=('col1', 'col3'), dtype=None)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["[1, 2]", "[1]", "{'a': 1, 'b': 2}", "[]", "{}"],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["['1', '2']", "['2']", "['1', '2', '3']", "['4', '5']", "['5', '4']"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────────────────┬──────┬─────────────────┬──────┐
    │ col1             ┆ col2 ┆ col3            ┆ col4 │
    │ ---              ┆ ---  ┆ ---             ┆ ---  │
    │ str              ┆ str  ┆ str             ┆ str  │
    ╞══════════════════╪══════╪═════════════════╪══════╡
    │ [1, 2]           ┆ 1    ┆ ['1', '2']      ┆ a    │
    │ [1]              ┆ 2    ┆ ['2']           ┆ b    │
    │ {'a': 1, 'b': 2} ┆ 3    ┆ ['1', '2', '3'] ┆ c    │
    │ []               ┆ 4    ┆ ['4', '5']      ┆ d    │
    │ {}               ┆ 5    ┆ ['5', '4']      ┆ e    │
    └──────────────────┴──────┴─────────────────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)

    ```
    """

    def __init__(self, columns: Sequence[str], dtype: type[pl.DataType] | None = None) -> None:
        self._columns = tuple(columns)
        self._dtype = dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(columns={self._columns}, dtype={self._dtype})"

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in tqdm(self._columns, desc="converting to JSON"):
            frame = frame.with_columns(
                frame.select(pl.col(col).str.replace_all("'", '"').str.json_decode(self._dtype))
            )
        return frame
