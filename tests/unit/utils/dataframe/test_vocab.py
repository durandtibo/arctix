from __future__ import annotations

from collections import Counter

import polars as pl
from coola import objects_are_equal

from arctix.utils.dataframe import generate_vocabulary
from arctix.utils.vocab import Vocabulary

#########################################
#     Tests for generate_vocabulary     #
#########################################


def test_generate_vocabulary_empty() -> None:
    assert objects_are_equal(
        generate_vocabulary(pl.DataFrame({"col": []}), col="col"), Vocabulary(Counter({}))
    )


def test_generate_vocabulary() -> None:
    assert objects_are_equal(
        generate_vocabulary(pl.DataFrame({"col": ["a", "b", "c", "d", "a"]}), col="col"),
        Vocabulary(Counter({"a": 2, "b": 1, "c": 1, "d": 1})),
    )
