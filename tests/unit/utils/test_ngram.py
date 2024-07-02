from __future__ import annotations

import pytest
from coola import objects_are_equal

from arctix.utils.ngram import find_ngrams

#################################
#     Tests for find_ngrams     #
#################################


def test_find_ngrams_n_1() -> None:
    assert objects_are_equal(
        list(find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=1)),
        [("a",), ("b",), ("c",), ("d",), ("e",), ("f",), ("g",), ("h",)],
    )


def test_find_ngrams_n_2() -> None:
    assert objects_are_equal(
        list(find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=2)),
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")],
    )


def test_find_ngrams_n_3() -> None:
    assert objects_are_equal(
        list(find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=3)),
        [
            ("a", "b", "c"),
            ("b", "c", "d"),
            ("c", "d", "e"),
            ("d", "e", "f"),
            ("e", "f", "g"),
            ("f", "g", "h"),
        ],
    )


def test_find_ngrams_empty() -> None:
    assert objects_are_equal(list(find_ngrams([], n=1)), [])


def test_find_ngrams_n_incorrect() -> None:
    with pytest.raises(RuntimeError, match="n must be greater or equal to 1"):
        find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=0)
