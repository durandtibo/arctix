from __future__ import annotations

from arctix.utils.format import str_indent, str_mapping

#################################
#     Tests for str_mapping     #
#################################


def test_str_mapping_empty() -> None:
    assert str_mapping({}) == ""


def test_str_mapping_1_item() -> None:
    assert str_mapping({"key": "value"}) == "(key): value"


def test_str_mapping_2_items() -> None:
    assert str_mapping({"key1": "value1", "key2": "value2"}) == "(key1): value1\n(key2): value2"


def test_str_mapping_sorted_values_true() -> None:
    assert (
        str_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1): value1\n(key2): value2"
    )


def test_str_mapping_sorted_values_false() -> None:
    assert str_mapping({"key2": "value2", "key1": "value1"}) == "(key2): value2\n(key1): value1"


################################
#     Tests for str_indent     #
################################


def test_str_indent_1_line() -> None:
    assert str_indent("abc") == "abc"


def test_str_indent_2_lines() -> None:
    assert str_indent("abc\n  def") == "abc\n    def"


def test_str_indent_num_spaces_2() -> None:
    assert str_indent("abc\ndef", num_spaces=2) == "abc\n  def"


def test_str_indent_num_spaces_4() -> None:
    assert str_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_str_indent_not_a_string() -> None:
    assert str_indent(123) == "123"
