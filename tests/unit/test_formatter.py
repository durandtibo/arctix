from pytest import mark, raises

from arctix import Summarizer
from arctix.formatter import DefaultFormatter

######################################
#     Tests for DefaultFormatter     #
######################################


def test_default_formatter_str() -> None:
    assert str(DefaultFormatter()).startswith("DefaultFormatter(")


def test_default_formatter_clone_max_characters_10() -> None:
    formatter = DefaultFormatter(max_characters=10)
    formatter_cloned = formatter.clone()
    formatter.set_max_characters(20)
    assert formatter is not formatter_cloned
    assert formatter.equal(DefaultFormatter(max_characters=20))
    assert formatter_cloned.equal(DefaultFormatter(max_characters=10))


def test_default_formatter_equal_true() -> None:
    assert DefaultFormatter().equal(DefaultFormatter())


def test_default_formatter_equal_false_different_max_characters() -> None:
    assert not DefaultFormatter().equal(DefaultFormatter(max_characters=10))


def test_default_formatter_equal_false_different_type() -> None:
    assert not DefaultFormatter().equal(42)


def test_default_formatter_format_str() -> None:
    assert DefaultFormatter().format(Summarizer(), "abc") == "<class 'str'> abc"


def test_default_formatter_format_int() -> None:
    assert DefaultFormatter().format(Summarizer(), 1) == "<class 'int'> 1"


def test_default_formatter_format_float() -> None:
    assert DefaultFormatter().format(Summarizer(), 1.2) == "<class 'float'> 1.2"


def test_default_formatter_format_max_characters_none() -> None:
    assert (
        DefaultFormatter(max_characters=None).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_format_max_characters_10() -> None:
    assert (
        DefaultFormatter(max_characters=10).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghij..."
    )


def test_default_formatter_format_max_characters_10_with_indent() -> None:
    assert (
        DefaultFormatter(max_characters=10).format(Summarizer(), "abc\tdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abc\tdefghi..."
    )


def test_default_formatter_format_max_characters_26() -> None:
    assert (
        DefaultFormatter(max_characters=26).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_format_max_characters_100() -> None:
    assert (
        DefaultFormatter(max_characters=100).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_load_state_dict() -> None:
    formatter = DefaultFormatter()
    formatter.load_state_dict({"max_characters": 10})
    assert formatter.equal(DefaultFormatter(max_characters=10))


def test_default_formatter_state_dict() -> None:
    assert DefaultFormatter().state_dict() == {"max_characters": None}


@mark.parametrize("max_characters", (1, 10, 100))
def test_default_formatter_set_max_characters_int(max_characters: int) -> None:
    formatter = DefaultFormatter()
    assert formatter._max_characters is None
    formatter.set_max_characters(max_characters)
    assert formatter._max_characters == max_characters


def test_default_formatter_set_max_characters_none() -> None:
    formatter = DefaultFormatter(max_characters=10)
    assert formatter._max_characters == 10
    formatter.set_max_characters(None)
    assert formatter._max_characters is None


def test_default_formatter_set_max_characters_incorrect_type() -> None:
    formatter = DefaultFormatter()
    with raises(TypeError, match="Incorrect type for max_characters. Expected int or None value"):
        formatter.set_max_characters(4.2)


@mark.parametrize("max_characters", (0, -1))
def test_default_formatter_set_max_characters_incorrect_value(max_characters: int) -> None:
    formatter = DefaultFormatter()
    with raises(
        ValueError, match="Incorrect value for max_characters. Expected a positive integer"
    ):
        formatter.set_max_characters(max_characters)
