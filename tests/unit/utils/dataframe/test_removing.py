from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.utils.dataframe import drop_duplicates

#####################################
#     Tests for drop_duplicates     #
#####################################


def test_drop_duplicates_empty() -> None:
    assert_frame_equal(drop_duplicates(pl.DataFrame({})), pl.DataFrame({}))


def test_drop_duplicates() -> None:
    assert_frame_equal(
        drop_duplicates(
            pl.DataFrame({"col1": [1, 2, 3, 1, 2, 3], "col2": [4, 5, 6, 4, 5, 6]}),
            maintain_order=True,
        ),
        pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
    )
