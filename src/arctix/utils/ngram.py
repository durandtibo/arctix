r"""Contain utility functions to compute n-grams."""

from __future__ import annotations

__all__ = ["find_ngrams", "find_seq_ngrams"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def find_ngrams(sequence: Sequence, n: int) -> list:
    r"""Find the n-grams of the input sequence.

    Args:
        sequence: The input sequence.
        n: The number of adjacent symbols.

    Returns:
        A list of n-grams.

    Raises:
        RuntimeError: if ``n`` is incorrect.

    Example usage:

    ```pycon

    >>> from arctix.utils.ngram import find_ngrams
    >>> seq = ["a", "b", "c", "d", "e", "f", "g", "h"]
    >>> ngrams = find_ngrams(seq, n=2)
    >>> ngrams
    [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'g'), ('g', 'h')]
    >>> ngrams = find_ngrams(seq, n=3)
    >>> ngrams
    [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e'), ('d', 'e', 'f'), ('e', 'f', 'g'), ('f', 'g', 'h')]

    ```
    """
    if n < 1:
        msg = f"n must be greater or equal to 1 but received {n}"
        raise RuntimeError(msg)
    return list(zip(*[sequence[i:] for i in range(n)]))


def find_seq_ngrams(seq_of_seqs: Sequence[Sequence], n: int) -> list:
    r"""Find the n-grams of a sequence of sequences.

    Args:
        seq_of_seqs: The input sequence of sequences.
        n: The number of adjacent symbols.

    Returns:
        A list of n-grams.

    Raises:
        RuntimeError: if ``n`` is incorrect.

    Example usage:

    ```pycon

    >>> from arctix.utils.ngram import find_seq_ngrams
    >>> seq = [["a", "b", "c", "d", "e"], ["f", "g", "h"]]
    >>> ngrams = find_seq_ngrams(seq, n=2)
    >>> ngrams
    [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('f', 'g'), ('g', 'h')]
    >>> ngrams = find_seq_ngrams(seq, n=3)
    >>> ngrams
    [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e'), ('f', 'g', 'h')]

    ```
    """
    if n < 1:
        msg = f"n must be greater or equal to 1 but received {n}"
        raise RuntimeError(msg)
    out = []
    for seq in seq_of_seqs:
        out.extend(find_ngrams(seq, n=n))
    return out
