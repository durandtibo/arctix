r"""Contain vocabulary-based transformers."""

from __future__ import annotations

__all__ = ["IndexToTokenDataFrameTransformer", "TokenToIndexDataFrameTransformer"]

from typing import TYPE_CHECKING, Any

import polars as pl

from arctix.transformer.dataframe.replace import ReplaceDataFrameTransformer

if TYPE_CHECKING:
    from arctix.utils.vocab import Vocabulary


class IndexToTokenDataFrameTransformer(ReplaceDataFrameTransformer):
    r"""Replace.

    Args:
        vocab: The vocabulary which contains the index to token
            mapping.
        index_column: The column name which contains the input indices.
        token_column: The column name which contains the output tokens.
        **kwargs: The keyword arguments to pass to ``replace``.

    Example usage:

    ```pycon

    >>> from collections import Counter
    >>> import polars as pl
    >>> from arctix.transformer.dataframe import IndexToToken
    >>> from arctix.utils.vocab import Vocabulary
    >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2, "d": 4}))
    >>> vocab.get_index_to_token()
    ('b', 'a', 'c', 'd')
    >>> transformer = IndexToToken(
    ...     vocab=vocab,
    ...     index_column="col",
    ...     token_column="token",
    ... )
    >>> transformer
    IndexToTokenDataFrameTransformer(orig_column=col, final_column=token)
    >>> frame = pl.DataFrame({"col": [1, 0, 2, 3, 1]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ col │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 0   │
    │ 2   │
    │ 3   │
    │ 1   │
    └─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬───────┐
    │ col ┆ token │
    │ --- ┆ ---   │
    │ i64 ┆ str   │
    ╞═════╪═══════╡
    │ 1   ┆ a     │
    │ 0   ┆ b     │
    │ 2   ┆ c     │
    │ 3   ┆ d     │
    │ 1   ┆ a     │
    └─────┴───────┘

    ```
    """

    def __init__(
        self,
        vocab: Vocabulary,
        index_column: str,
        token_column: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            orig_column=index_column,
            final_column=token_column,
            old=list(range(len(vocab))),
            new=vocab.get_index_to_token(),
            **kwargs,
        )


class TokenToIndexDataFrameTransformer(ReplaceDataFrameTransformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        vocab: The vocabulary which contains the token to index
            mapping.
        token_column: The column name which contains the input tokens.
        index_column: The column name which contains the output
            indices.
        **kwargs: The keyword arguments to pass to ``replace``.

    Example usage:

    ```pycon

    >>> from collections import Counter
    >>> import polars as pl
    >>> from arctix.transformer.dataframe import TokenToIndex
    >>> from arctix.utils.vocab import Vocabulary
    >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2, "d": 4}))
    >>> vocab.get_token_to_index()
    {'b': 0, 'a': 1, 'c': 2, 'd': 3}
    >>> transformer = TokenToIndex(
    ...     vocab=vocab, token_column="col", index_column="index"
    ... )
    >>> transformer
    TokenToIndexDataFrameTransformer(orig_column=col, final_column=index)
    >>> frame = pl.DataFrame({"col": ["a", "b", "c", "d", "a"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ col │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    │ d   │
    │ a   │
    └─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬───────┐
    │ col ┆ index │
    │ --- ┆ ---   │
    │ str ┆ i64   │
    ╞═════╪═══════╡
    │ a   ┆ 1     │
    │ b   ┆ 0     │
    │ c   ┆ 2     │
    │ d   ┆ 3     │
    │ a   ┆ 1     │
    └─────┴───────┘

    ```
    """

    def __init__(
        self,
        vocab: Vocabulary,
        token_column: str,
        index_column: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            orig_column=token_column,
            final_column=index_column,
            old=vocab.get_token_to_index(),
            **(kwargs or {"return_dtype": pl.Int64}),
        )
