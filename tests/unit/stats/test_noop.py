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
