from __future__ import annotations

from arctix.utils.mapping import convert_to_dict_of_flat_lists

###################################################
#     Tests for convert_to_dict_of_flat_lists     #
###################################################


def test_convert_to_dict_of_flat_lists_empty_list() -> None:
    assert convert_to_dict_of_flat_lists([]) == {}


def test_convert_to_dict_of_flat_lists_empty_dict() -> None:
    assert convert_to_dict_of_flat_lists([{}]) == {}


def test_convert_to_dict_of_flat_lists() -> None:
    assert convert_to_dict_of_flat_lists(
        [
            {"key1": [1, 2], "key2": [10, 11]},
            {"key1": [2], "key2": [20]},
            {"key1": [3, 4, 5], "key2": [30, 31, 32]},
        ]
    ) == {"key1": [1, 2, 2, 3, 4, 5], "key2": [10, 11, 20, 30, 31, 32]}
