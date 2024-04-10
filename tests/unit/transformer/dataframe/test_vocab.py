from __future__ import annotations

from collections import Counter

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import IndexToToken, TokenToIndex
from arctix.utils.vocab import Vocabulary


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return Vocabulary(Counter({"b": 3, "a": 1, "c": 2, "d": 4}))


######################################################
#     Tests for IndexToTokenDataFrameTransformer     #
######################################################


def test_index_to_token_dataframe_transformer_repr(vocab: Vocabulary) -> None:
    assert repr(
        IndexToToken(
            vocab=vocab,
            index_column="index",
            token_column="token",  # noqa: S106
        )
    ).startswith("IndexToTokenDataFrameTransformer(")


def test_index_to_token_dataframe_transformer_str(vocab: Vocabulary) -> None:
    assert str(
        IndexToToken(
            vocab=vocab,
            index_column="index",
            token_column="token",  # noqa: S106
        )
    ).startswith("IndexToTokenDataFrameTransformer(")


def test_index_to_token_dataframe_transformer_transform(vocab: Vocabulary) -> None:
    transformer = IndexToToken(
        vocab=vocab,
        index_column="index",
        token_column="token",  # noqa: S106
    )
    frame = pl.DataFrame({"index": [1, 0, 2, 3, 1]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"index": [1, 0, 2, 3, 1], "token": ["a", "b", "c", "d", "a"]})
    )


######################################################
#     Tests for TokenToIndexDataFrameTransformer     #
######################################################


def test_token_to_index_dataframe_transformer_repr(vocab: Vocabulary) -> None:
    assert repr(
        TokenToIndex(
            vocab=vocab,
            token_column="token",  # noqa: S106
            index_column="index",
        )
    ).startswith("TokenToIndexDataFrameTransformer(")


def test_token_to_index_dataframe_transformer_str(vocab: Vocabulary) -> None:
    assert str(
        TokenToIndex(
            vocab=vocab,
            token_column="token",  # noqa: S106
            index_column="index",
        )
    ).startswith("TokenToIndexDataFrameTransformer(")


def test_token_to_index_dataframe_transformer_transform(vocab: Vocabulary) -> None:
    transformer = TokenToIndex(
        vocab=vocab,
        token_column="token",  # noqa: S106
        index_column="index",
    )
    frame = pl.DataFrame({"token": ["a", "b", "c", "d", "a"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"token": ["a", "b", "c", "d", "a"], "index": [1, 0, 2, 3, 1]})
    )
