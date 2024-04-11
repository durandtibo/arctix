from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Sort

##############################################
#     Tests for SortDataFrameTransformer     #
##############################################


def test_sort_dataframe_transformer_str() -> None:
    assert str(Sort(columns=["col3", "col1"])).startswith("SortDataFrameTransformer(")


def test_sort_dataframe_transformer_transform() -> None:
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 1, None, 2],
                "col2": [None, 6.0, 4.0, 5.0],
                "col3": [None, "a", "b", "c"],
            }
        ),
    )


def test_sort_dataframe_transformer_transform_null_last() -> None:
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    transformer = Sort(columns=["col3", "col1"], nulls_last=True)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, None, 2, None],
                "col2": [6.0, 4.0, 5.0, None],
                "col3": ["a", "b", "c", None],
            }
        ),
    )
