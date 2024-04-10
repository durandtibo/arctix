r"""Contain some utility functions to manipulate DataFrames."""

from __future__ import annotations

__all__ = ["drop_duplicates", "generate_vocabulary"]

from arctix.utils.dataframe.removing import drop_duplicates
from arctix.utils.dataframe.vocab import generate_vocabulary
