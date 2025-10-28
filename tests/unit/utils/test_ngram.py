from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from coola import objects_are_equal

from arctix.utils.imports import is_matplotlib_available, matplotlib_available
from arctix.utils.ngram import (
    _create_transition_matrix,
    find_ngrams,
    find_seq_ngrams,
    plot_ngrams,
)
from arctix.utils.vocab import Vocabulary

if is_matplotlib_available():
    import matplotlib.pyplot as plt


#################################
#     Tests for find_ngrams     #
#################################


def test_find_ngrams_n_1() -> None:
    assert objects_are_equal(
        find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=1),
        [("a",), ("b",), ("c",), ("d",), ("e",), ("f",), ("g",), ("h",)],
    )


def test_find_ngrams_n_2() -> None:
    assert objects_are_equal(
        find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=2),
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")],
    )


def test_find_ngrams_n_3() -> None:
    assert objects_are_equal(
        find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=3),
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
    assert objects_are_equal(find_ngrams([], n=1), [])


def test_find_ngrams_n_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"n must be greater or equal to 1"):
        find_ngrams(["a", "b", "c", "d", "e", "f", "g", "h"], n=0)


#####################################
#     Tests for find_seq_ngrams     #
#####################################


def test_find_seq_ngrams_n_1() -> None:
    assert objects_are_equal(
        find_seq_ngrams([["a", "b", "c", "d", "e"], ["f", "g", "h"], ["i"]], n=1),
        [("a",), ("b",), ("c",), ("d",), ("e",), ("f",), ("g",), ("h",), ("i",)],
    )


def test_find_seq_ngrams_n_2() -> None:
    assert objects_are_equal(
        find_seq_ngrams([["a", "b", "c", "d", "e"], ["f", "g", "h"], ["i"]], n=2),
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("f", "g"), ("g", "h")],
    )


def test_find_seq_ngrams_n_3() -> None:
    assert objects_are_equal(
        find_seq_ngrams([["a", "b", "c", "d", "e"], ["f", "g", "h"], ["i"]], n=3),
        [("a", "b", "c"), ("b", "c", "d"), ("c", "d", "e"), ("f", "g", "h")],
    )


def test_find_seq_ngrams_empty() -> None:
    assert objects_are_equal(find_seq_ngrams([], n=1), [])


def test_find_seq_ngrams_n_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"n must be greater or equal to 1"):
        find_seq_ngrams([["a", "b", "c", "d", "e"], ["f", "g", "h"], ["i"]], n=0)


#################################
#     Tests for plot_ngrams     #
#################################


@matplotlib_available
def test_plot_ngrams() -> None:
    _fig, ax = plt.subplots(figsize=(6, 6))
    plot_ngrams(
        ngrams=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")],
        ax=ax,
    )


@matplotlib_available
def test_plot_ngrams_empty() -> None:
    _fig, ax = plt.subplots(figsize=(6, 6))
    plot_ngrams(ngrams=[], ax=ax)


##############################################
#     Tests for create_transition_matrix     #
##############################################


def test_create_transition_matrix() -> None:
    assert objects_are_equal(
        _create_transition_matrix(
            counter=Counter([("a", "1"), ("b", "2"), ("c", "3"), ("e", "2")]),
            previous_vocab=Vocabulary(Counter([("a",), ("b",), ("c",), ("d",), ("e",)])),
            next_vocab=Vocabulary(Counter(["1", "2", "3"])),
        ),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=np.float64),
    )
