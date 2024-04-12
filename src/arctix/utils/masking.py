r"""Contain utility functions to generate or manipulate masks."""

from __future__ import annotations

__all__ = ["convert_sequences_to_array", "generate_mask_from_lengths"]

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike


def convert_sequences_to_array(
    data: Sequence[Sequence], max_len: int, dtype: DTypeLike = None, padded_value: Any = 0
) -> np.ndarray:
    r"""Convert a list of sequences to a ``numpy.ndarray``.

    The sequences can have different lengths, so the sequences are
    padded to all have the same length.

    Args:
        data: The list of sequence to convert to an array.
        max_len: The maximum sequence length which is used to define
            the second dimension of the array. If a sequence is longer,
            it is truncated to the maximum sequence length.
        dtype: The data type of the generated array.
        padded_value: The value used to pad the sequences.

    Returns:
        The generated array.

    Example usage:

    ```pycon

    >>> from arctix.utils.masking import convert_sequences_to_array
    >>> arr = convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=5)
    >>> arr
    array([[1, 2, 3, 0, 0],
           [9, 8, 7, 6, 5],
           [1, 0, 0, 0, 0]])
    >>> arr = convert_sequences_to_array(
    ...     [[1, 2, 3], [9, 8, 7, 6, 5], [1]],
    ...     max_len=5,
    ...     dtype=float,
    ...     padded_value=-1,
    ... )
    >>> arr
    array([[ 1.,  2.,  3., -1., -1.],
           [ 9.,  8.,  7.,  6.,  5.],
           [ 1., -1., -1., -1., -1.]])

    ```
    """
    array = np.full((len(data), max_len), fill_value=padded_value, dtype=dtype)
    for i, values in enumerate(data):
        length = min(len(values), max_len)
        array[i, :length] = values[:length]
    return array


def generate_mask_from_lengths(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    r"""Generate a mask from the sequences lengths.

    The mask indicates masked values. ``True`` indicates a
    masked/invalid value and ``False`` indicates a valid value.

    Args:
        lengths: The lengths of each sequence in the batch.
        max_len: The maximum sequence length. If ``None``, the maximum
            length is computed based on the given lengths.

    Returns:
        The generated mask of shape ``(batch_size, max_len)``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arctix.utils.masking import generate_mask_from_lengths
    >>> mask = generate_mask_from_lengths(lengths=np.array([4, 3, 5, 3, 2]))
    >>> mask
    array([[False, False, False, False,  True],
           [False, False, False,  True,  True],
           [False, False, False, False, False],
           [False, False, False,  True,  True],
           [False, False,  True,  True,  True]])

    ```
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = lengths.max(initial=0)
    lengths = np.broadcast_to(lengths.reshape(batch_size, 1), (batch_size, max_len))
    indices = np.broadcast_to(np.arange(max_len).reshape(1, max_len), (batch_size, max_len))
    return indices >= lengths
