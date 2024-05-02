from __future__ import annotations

import datetime

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import TimeToSecond

######################################################
#     Tests for TimeToSecondDataFrameTransformer     #
######################################################


def test_time_to_second_dataframe_transformer_str() -> None:
    assert str(TimeToSecond(in_col="time", out_col="second")).startswith(
        "TimeToSecondDataFrameTransformer("
    )


def test_time_to_second_dataframe_transformer_transform() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(0, 0, 1, 890000),
                datetime.time(0, 1, 1, 890000),
                datetime.time(1, 1, 1, 890000),
                datetime.time(0, 19, 19, 890000),
                datetime.time(19, 19, 19, 420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="time", out_col="second")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "time": [
                    datetime.time(0, 0, 1, 890000),
                    datetime.time(0, 1, 1, 890000),
                    datetime.time(1, 1, 1, 890000),
                    datetime.time(0, 19, 19, 890000),
                    datetime.time(19, 19, 19, 420000),
                ],
                "col": ["a", "b", "c", "d", "e"],
                "second": [1.89, 61.89, 3661.89, 1159.89, 69559.42],
            },
            schema={"time": pl.Time, "col": pl.String, "second": pl.Float64},
        ),
    )
