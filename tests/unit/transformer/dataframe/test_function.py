from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Function

##################################################
#     Tests for FunctionDataFrameTransformer     #
##################################################


def my_filter(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("col1").is_in({2, 4}))


def test_function_dataframe_transformer_repr() -> None:
    assert repr(Function(func=my_filter)).startswith("FunctionDataFrameTransformer(")


def test_function_dataframe_transformer_str() -> None:
    assert str(Function(func=my_filter)).startswith("FunctionDataFrameTransformer(")


def test_function_dataframe_transformer_transform() -> None:
    transformer = Function(func=my_filter)
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
        }
    )
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col1": [2, 4], "col2": ["2", "4"]}))
