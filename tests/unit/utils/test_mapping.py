from __future__ import annotations

from coola import objects_are_equal

from arctix.utils.mapping import (
    convert_to_dict_of_flat_lists,
    sort_by_key,
    sort_by_value,
)

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


#################################
#     Tests for sort_by_key     #
#################################


def test_sort_by_key() -> None:
    assert objects_are_equal(sort_by_key({"b": 3, "c": 1, "a": 2}), {"a": 2, "b": 3, "c": 1})


def test_sort_by_key_reverse() -> None:
    assert objects_are_equal(
        sort_by_key({"b": 3, "c": 1, "a": 2}, reverse=True), {"c": 1, "b": 3, "a": 2}
    )


def test_sort_by_key_empty() -> None:
    assert objects_are_equal(sort_by_key({}), {})


###################################
#     Tests for sort_by_value     #
###################################


def test_sort_by_value() -> None:
    assert objects_are_equal(
        sort_by_value({"b": 3, "c": 1, "a": 2, "d": 2}), {"c": 1, "a": 2, "d": 2, "b": 3}
    )


def test_sort_by_value_reverse() -> None:
    assert objects_are_equal(
        sort_by_value({"b": 3, "c": 1, "a": 2, "d": 2}, reverse=True),
        {"b": 3, "a": 2, "d": 2, "c": 1},
    )


def test_sort_by_value_empty() -> None:
    assert objects_are_equal(sort_by_value({}), {})
