from __future__ import annotations

import numpy as np

from arctix.utils.masking import generate_mask_from_lengths

################################################
#     Tests for generate_mask_from_lengths     #
################################################


def test_generate_mask_from_lengths() -> None:
    assert np.array_equal(
        generate_mask_from_lengths(np.array([4, 3, 5, 3, 2])),
        np.array(
            [
                [False, False, False, False, True],
                [False, False, False, True, True],
                [False, False, False, False, False],
                [False, False, False, True, True],
                [False, False, True, True, True],
            ]
        ),
    )


def test_generate_mask_from_lengths_max_len_4() -> None:
    assert np.array_equal(
        generate_mask_from_lengths(np.array([4, 3, 5, 3, 2]), max_len=4),
        np.array(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, False, False, False],
                [False, False, False, True],
                [False, False, True, True],
            ]
        ),
    )


def test_generate_mask_from_lengths_max_len_6() -> None:
    assert np.array_equal(
        generate_mask_from_lengths(np.array([4, 3, 5, 3, 2]), max_len=6),
        np.array(
            [
                [False, False, False, False, True, True],
                [False, False, False, True, True, True],
                [False, False, False, False, False, True],
                [False, False, False, True, True, True],
                [False, False, True, True, True, True],
            ]
        ),
    )


def test_generate_mask_from_lengths_empty() -> None:
    assert np.array_equal(
        generate_mask_from_lengths(np.array([], dtype=int)), np.zeros((0, 0), dtype=bool)
    )


def test_generate_mask_from_lengths_batch_size_1() -> None:
    assert np.array_equal(
        generate_mask_from_lengths(np.array([4])), np.array([[False, False, False, False]])
    )
