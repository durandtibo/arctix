r"""Contain utility functions to compute n-grams."""

from __future__ import annotations

__all__ = ["find_ngrams"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def find_ngrams(sequence: Sequence, n: int) -> Iterator:
    r"""Find the n-grams of the input sequence.

    Args:
        sequence: The input sequence.
        n: The number of adjacent symbols.

    Returns:
        An iterator on the n-grams.

    Raises:
        RuntimeError: if ``n`` is incorrect.

    Example usage:

    ```pycon

    >>> from arctix.utils.ngram import find_ngrams
    >>> seq = ["a", "b", "c", "d", "e", "f", "g", "h"]
    >>> ngrams = list(find_ngrams(seq, n=2))
    >>> ngrams
    [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'g'), ('g', 'h')]
    >>> ngrams = list(find_ngrams(seq, n=3))
    >>> ngrams
    [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e'), ('d', 'e', 'f'), ('e', 'f', 'g'), ('f', 'g', 'h')]

    ```
    """
    if n < 1:
        msg = f"n must be greater or equal to 1 but received {n}"
        raise RuntimeError(msg)
    return zip(*[sequence[i:] for i in range(n)])
