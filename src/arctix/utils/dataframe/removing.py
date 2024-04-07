r"""Contain some utility functions to remove items in DataFrames."""

from __future__ import annotations

__all__ = ["drop_duplicates"]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def drop_duplicates(frame: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
    r"""Return DataFrame with duplicate rows removed.

    Args:
        frame: The input DataFrame.
        *args: Positional arguments that are passed to
            ``polars.DataFrame.unique``.
        **kwargs: Keyword arguments that are passed to
            ``polars.DataFrame.unique``.

    Returns:
        The DataFrame with duplicate rows removed.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.utils.dataframe import drop_duplicates
    >>> df = pl.DataFrame({"col1": [1, 2, 3, 1, 2, 3], "col2": [4, 5, 6, 4, 5, 6]})
    >>> out = drop_duplicates(df, keep="first", maintain_order=True)
    >>> out
    shape: (3, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ 1    ┆ 4    │
    │ 2    ┆ 5    │
    │ 3    ┆ 6    │
    └──────┴──────┘

    ```
    """
    orig_shape = frame.shape
    if orig_shape[0] == 0:
        return frame
    logger.info("removing duplicate rows...")
    out = frame.unique(*args, **kwargs)
    logger.info(f"removed {orig_shape[0] - out.shape[0]:,} rows | {orig_shape} -> {out.shape}")
    return out
