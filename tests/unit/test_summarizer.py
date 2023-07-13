from typing import Any
from unittest.mock import Mock

from pytest import fixture, mark

from arctix import (
    BaseSummarizer,
    Summarizer,
    set_summarizer_options,
    summarizer_options,
    summary,
)


@fixture(autouse=True, scope="function")
def reset() -> None:
    state = Summarizer.state_dict()
    try:
        yield
    finally:
        Summarizer.load_state_dict(state)


#############################
#     Tests for summary     #
#############################


def test_summary_bool() -> None:
    assert summary(True) == "<class 'bool'> True"


def test_summary_int() -> None:
    assert summary(42) == "<class 'int'> 42"


def test_summary_float() -> None:
    assert summary(4.2) == "<class 'float'> 4.2"


def test_summary_dict() -> None:
    assert summary({"key": "value"}) == "<class 'dict'> (length=1)\n  (key): value"


def test_summary_list() -> None:
    assert summary(["abc", "def"]) == "<class 'list'> (length=2)\n  (0): abc\n  (1): def"


def test_summary_tuple() -> None:
    assert summary(("abc", "def")) == "<class 'tuple'> (length=2)\n  (0): abc\n  (1): def"


def test_summary_max_depth_1() -> None:
    assert (
        summary([[0, 1, 2], {"key1": "abc", "key2": "def"}])
        == "<class 'list'> (length=2)\n  (0): [0, 1, 2]\n  (1): {'key1': 'abc', 'key2': 'def'}"
    )


def test_summary_max_depth_2() -> None:
    assert summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): 0\n"
        "      (1): 1\n"
        "      (2): 2\n"
        "  (1): <class 'dict'> (length=2)\n"
        "      (key1): abc\n"
        "      (key2): def"
    )


def test_summary_max_depth_3() -> None:
    assert summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=3) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): <class 'int'> 0\n"
        "      (1): <class 'int'> 1\n"
        "      (2): <class 'int'> 2\n"
        "  (1): <class 'dict'> (length=2)\n"
        "      (key1): <class 'str'> abc\n"
        "      (key2): <class 'str'> def"
    )


@mark.parametrize("max_depth", (0, -1, -2))
def test_summary_max_depth_0(max_depth: int) -> None:
    assert (
        summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=max_depth)
        == "[[0, 1, 2], {'key1': 'abc', 'key2': 'def'}]"
    )


@mark.parametrize("value", ("abc", 42))
@mark.parametrize("max_depth", (1, 2))
def test_summary_summarizer(value: Any, max_depth: int) -> None:
    summarizer = Mock(spec=BaseSummarizer)
    summary(value, max_depth=max_depth, summarizer=summarizer)
    summarizer.summary.assert_called_once_with(value=value, depth=0, max_depth=max_depth)


############################################
#     Tests for set_summarizer_options     #
############################################


def test_set_summarizer_options_empty() -> None:
    state = Summarizer.state_dict()
    set_summarizer_options()
    assert state == Summarizer.state_dict()


@mark.parametrize("max_characters", (-1, 0, 1, 10))
def test_set_summarizer_options_max_characters_int(max_characters: int) -> None:
    set_summarizer_options(max_characters=max_characters)
    assert Summarizer.registry[object].get_max_characters() == max_characters


########################################
#     Tests for summarizer_options     #
########################################


def test_summarizer_options_empty() -> None:
    state = Summarizer.state_dict()
    with summarizer_options():
        assert state == Summarizer.state_dict()


def test_summarizer_options_max_characters_none() -> None:
    state = Summarizer.state_dict()
    with summarizer_options(max_characters=None):
        assert state == Summarizer.state_dict()


@mark.parametrize("max_characters", (-1, 0, 1, 10))
def test_summarizer_options_max_characters_int(max_characters: int) -> None:
    assert Summarizer.registry[object].get_max_characters() == -1
    with summarizer_options(max_characters=max_characters):
        assert Summarizer.registry[object].get_max_characters() == max_characters
    assert Summarizer.registry[object].get_max_characters() == -1


# TODO: add set
