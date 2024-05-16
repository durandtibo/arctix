from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Diff

##############################################
#     Tests for DiffDataFrameTransformer     #
##############################################


def test_diff_dataframe_transformer_str() -> None:
    assert str(Diff(in_col="col1", out_col="diff")).startswith("DiffDataFrameTransformer(")


def test_diff_dataframe_transformer_transform_int32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, 1, 1, 1, 1],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "diff": pl.Int64},
        ),
    )


def test_diff_dataframe_transformer_transform_float32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Float32, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, 1.0, 1.0, 1.0, 1.0],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "diff": pl.Float32},
        ),
    )
