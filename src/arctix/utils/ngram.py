r"""Contain utility functions to compute n-grams."""

from __future__ import annotations

__all__ = ["find_ngrams", "find_seq_ngrams", "plot_ngrams"]

from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from arctix.utils.imports import is_matplotlib_available
from arctix.utils.vocab import Vocabulary

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_matplotlib_available():
    import matplotlib.pyplot as plt
else:  # pragma: no cover
    plt = Mock()


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


def plot_ngrams(ngrams: Sequence, ax: plt.Axes) -> None:
    r"""Plot the transition matrix of the n-grams.

    Args:
        ngrams: The sequence of n-grams.
        ax: The matplotlib axes to use to plot the transition.

    Example usage:

    ```pycon

    >>> import matplotlib.pyplot as plt
    >>> from arctix.utils.ngram import plot_ngrams
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> plot_ngrams(
    ...     ngrams=[
    ...         ("a", "b"),
    ...         ("b", "c"),
    ...         ("c", "d"),
    ...         ("d", "e"),
    ...         ("e", "f"),
    ...         ("f", "g"),
    ...         ("g", "h"),
    ...     ],
    ...     ax=ax,
    ... )

    ```
    """
    if not ngrams:
        return

    counter = Counter(ngrams)
    previous_vocab = Vocabulary(Counter([tokens[:-1] for tokens in counter])).sort_by_token()
    next_vocab = Vocabulary(Counter([tokens[-1] for tokens in counter])).sort_by_token()
    transition = _create_transition_matrix(
        counter=counter, previous_vocab=previous_vocab, next_vocab=next_vocab
    )

    ax.imshow(transition)
    ax.set_yticks(np.arange(len(previous_vocab)), labels=previous_vocab.get_index_to_token())
    ax.set_xticks(np.arange(len(next_vocab)), labels=next_vocab.get_index_to_token())
    ax.tick_params(axis="x", labelrotation=90)


def _create_transition_matrix(
    counter: Counter, previous_vocab: Vocabulary, next_vocab: Vocabulary
) -> np.ndarray:
    r"""Create the transition matrix between the previous token(s) and
    the next token.

    Args:
        counter: The counter associated to n-grams.
        previous_vocab: The vocabulary of previous tokens.
        next_vocab: The vocabulary of next tokens.

    Returns:
        The transition matrix between the previous token(s) and the
            next token.
    """
    transition = np.zeros((len(previous_vocab), len(next_vocab)))
    for tokens, count in counter.items():
        prev_tokens = previous_vocab.get_index(tokens[:-1])
        next_token = next_vocab.get_index(tokens[-1])
        transition[prev_tokens, next_token] += count
    return transition
