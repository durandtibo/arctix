from __future__ import annotations

import numpy as np

from arctix.utils.masking import convert_sequences_to_array, generate_mask_from_lengths

################################################
#     Tests for convert_sequences_to_array     #
################################################


def test_convert_sequences_to_array() -> None:
    assert np.array_equal(
        convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=5),
        np.array([[1, 2, 3, 0, 0], [9, 8, 7, 6, 5], [1, 0, 0, 0, 0]]),
    )


def test_convert_sequences_to_array_max_len_4() -> None:
    assert np.array_equal(
        convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=4),
        np.array([[1, 2, 3, 0], [9, 8, 7, 6], [1, 0, 0, 0]]),
    )


def test_convert_sequences_to_array_max_len_6() -> None:
    assert np.array_equal(
        convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=6),
        np.array([[1, 2, 3, 0, 0, 0], [9, 8, 7, 6, 5, 0], [1, 0, 0, 0, 0, 0]]),
    )


def test_convert_sequences_to_array_dtype() -> None:
    assert np.array_equal(
        convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=5, dtype=float),
        np.array([[1.0, 2.0, 3.0, 0.0, 0.0], [9.0, 8.0, 7.0, 6.0, 5.0], [1.0, 0.0, 0.0, 0.0, 0.0]]),
    )


def test_convert_sequences_to_array_padded_value() -> None:
    assert np.array_equal(
        convert_sequences_to_array([[1, 2, 3], [9, 8, 7, 6, 5], [1]], max_len=5, padded_value=-1),
        np.array([[1, 2, 3, -1, -1], [9, 8, 7, 6, 5], [1, -1, -1, -1, -1]]),
    )


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
