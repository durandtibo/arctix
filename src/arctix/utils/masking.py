r"""Contain utility functions to generate or manipulate masks."""

from __future__ import annotations

__all__ = ["generate_mask_from_lengths"]

import numpy as np


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
