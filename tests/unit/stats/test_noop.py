from typing import Any

from pytest import mark

from arctix.stats import NoOpTracker

#################################
#     Tests for NoOpTracker     #
#################################


def test_noop_tracker_str() -> None:
    assert str(NoOpTracker()).startswith("NoOpTracker(")


@mark.parametrize("data", (1, "abc"))
def test_noop_tracker_add(data: Any) -> None:
    tracker = NoOpTracker()
    tracker.add(data)  # check it does not raise error


def test_noop_tracker_get_statistics() -> None:
    tracker = NoOpTracker()
    assert tracker.get_statistics() == {}


def test_noop_tracker_reset() -> None:
    tracker = NoOpTracker()
    tracker.add(1)
    tracker.reset()  # check it does not raise error


def test_noop_tracker_load_state_dict() -> None:
    tracker = NoOpTracker()
    tracker.load_state_dict({})
    assert tracker.state_dict() == {}


def test_noop_tracker_state_dict() -> None:
    tracker = NoOpTracker()
    tracker.add(1)
    assert tracker.state_dict() == {}


def test_noop_tracker_state_dict_empty() -> None:
    assert NoOpTracker().state_dict() == {}
