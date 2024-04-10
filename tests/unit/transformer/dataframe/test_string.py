from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import StripChars

####################################################
#     Tests for StripCharsDataFrameTransformer     #
####################################################


def test_strip_chars_dataframe_transformer_repr() -> None:
    assert repr(StripChars(columns=["col1", "col3"])).startswith(
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'))"
    )


def test_strip_chars_dataframe_transformer_str() -> None:
    assert str(StripChars(columns=["col1", "col3"])).startswith(
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'))"
    )


def test_strip_chars_dataframe_transformer_transform() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    transformer = StripChars(columns=["col2", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
            }
        ),
    )


def test_strip_chars_dataframe_transformer_transform_none() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None],
            "col2": ["1", "2", "3", "4", "5", None],
            "col3": ["a ", " b", "  c  ", "d", "e", None],
            "col4": ["a ", " b", "  c  ", "d", "e", None],
        }
    )
    transformer = StripChars(columns=["col2", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": ["1", "2", "3", "4", "5", None],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": ["a ", " b", "  c  ", "d", "e", None],
            }
        ),
    )


def test_strip_chars_dataframe_transformer_transform_empty() -> None:
    transformer = StripChars(columns=[])
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))
