from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Replace

#################################################
#     Tests for ReplaceDataFrameTransformer     #
#################################################


def test_replace_dataframe_transformer_repr() -> None:
    assert repr(
        Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceDataFrameTransformer(")


def test_replace_dataframe_transformer_str() -> None:
    assert str(
        Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceDataFrameTransformer(")


def test_replace_dataframe_transformer_transform_mapping() -> None:
    transformer = Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": ["1", "2", "3", "d", "e"]})
    )


def test_replace_dataframe_transformer_transform_mapping_default() -> None:
    transformer = Replace(
        orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3}, default=None
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, None, None]})
    )


def test_replace_dataframe_transformer_transform_same_column() -> None:
    transformer = Replace(orig_column="col", final_column="col", old={"a": "1", "b": "2", "c": "3"})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": ["1", "2", "3", "d", "e"]}))
