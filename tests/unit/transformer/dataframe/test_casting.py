from __future__ import annotations

from datetime import time

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Cast, ToTime

##############################################
#     Tests for CastDataFrameTransformer     #
##############################################


def test_cast_dataframe_transformer_str() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32)).startswith(
        "CastDataFrameTransformer("
    )


def test_cast_dataframe_transformer_transform_int32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.String, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_cast_dataframe_transformer_transform_float32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = Cast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "col4": pl.String},
        ),
    )


################################################
#     Tests for ToTimeDataFrameTransformer     #
################################################


def test_to_time_dataframe_transformer_str() -> None:
    assert str(ToTime(columns=["col1", "col3"])).startswith("ToTimeDataFrameTransformer(")


def test_to_time_dataframe_transformer_transform_no_format() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    time(1, 1, 1),
                    time(2, 2, 2),
                    time(12, 0, 1),
                    time(18, 18, 18),
                    time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    time(11, 11, 11),
                    time(12, 12, 12),
                    time(13, 13, 13),
                    time(8, 8, 8),
                    time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_dataframe_transformer_transform_format() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3"], format="%H:%M:%S")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    time(1, 1, 1),
                    time(2, 2, 2),
                    time(12, 0, 1),
                    time(18, 18, 18),
                    time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    time(11, 11, 11),
                    time(12, 12, 12),
                    time(13, 13, 13),
                    time(8, 8, 8),
                    time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )
