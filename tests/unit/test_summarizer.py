from pytest import fixture, mark

from arctix import Summarizer, set_summarizer_options, summarizer_options


@fixture(autouse=True, scope="function")
def reset() -> None:
    state = Summarizer.state_dict()
    try:
        yield
    finally:
        Summarizer.load_state_dict(state)


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
