from __future__ import annotations

from unittest.mock import Mock

from coola import objects_are_allclose, objects_are_equal
from coola.utils import is_numpy_available, is_torch_available
from pytest import mark, raises

from arctix.stats import (
    AutoTracker,
    ContinuousTracker,
    DiscreteTracker,
    EmptyTrackerError,
)
from arctix.testing import numpy_available, torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()

#################################
#     Tests for AutoTracker     #
#################################


def test_auto_tracker_str() -> None:
    assert str(AutoTracker()).startswith("AutoTracker(")


def test_auto_tracker_add_bool() -> None:
    tracker = AutoTracker()
    tracker.add(True)
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {True: 1})


def test_auto_tracker_add_int() -> None:
    tracker = AutoTracker()
    tracker.add(1)
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (1,))


def test_auto_tracker_add_float() -> None:
    tracker = AutoTracker()
    tracker.add(4.2)
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (4.2,))


def test_auto_tracker_add_str() -> None:
    tracker = AutoTracker()
    tracker.add("meow")
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {"meow": 1})


def test_auto_tracker_add_seq_float() -> None:
    tracker = AutoTracker()
    tracker.add([4.2, 1.0, 2.2])
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (4.2, 1.0, 2.2))


def test_auto_tracker_add_seq_int() -> None:
    tracker = AutoTracker()
    tracker.add([1, 2, 4])
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (1, 2, 4))


def test_auto_tracker_add_seq_mix() -> None:
    tracker = AutoTracker()
    tracker.add([1, 2, "meow"])
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {1: 1, 2: 1, "meow": 1})


@numpy_available
def test_auto_tracker_add_ndarray_bool() -> None:
    tracker = AutoTracker()
    tracker.add(np.array([True, False, True, True]))
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {True: 3, False: 1})


@numpy_available
def test_auto_tracker_add_ndarray_float() -> None:
    tracker = AutoTracker()
    tracker.add(np.array([4.2, 1.0, 2.2], dtype=float))
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_allclose(tracker.tracker.values, (4.2, 1.0, 2.2), atol=1e-6)


@numpy_available
def test_auto_tracker_add_ndarray_int() -> None:
    tracker = AutoTracker()
    tracker.add(np.array([1, 2, 4], dtype=int))
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (1, 2, 4))


@torch_available
def test_auto_tracker_add_tensor_bool() -> None:
    tracker = AutoTracker()
    tracker.add(torch.tensor([True, False, True, True]))
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {True: 3, False: 1})


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([4.2, 1.0, 2.2], dtype=torch.float),
        torch.tensor([4.2, 1.0, 2.2], dtype=torch.double),
    ),
)
def test_auto_tracker_add_tensor_float(tensor: torch.Tensor) -> None:
    tracker = AutoTracker()
    tracker.add(tensor)
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_allclose(tracker.tracker.values, (4.2, 1.0, 2.2), atol=1e-6)


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([1, 2, 4], dtype=torch.int),
        torch.tensor([1, 2, 4], dtype=torch.long),
    ),
)
def test_auto_tracker_add_tensor_int(tensor: torch.Tensor) -> None:
    tracker = AutoTracker()
    tracker.add(tensor)
    assert isinstance(tracker.tracker, ContinuousTracker)
    assert objects_are_equal(tracker.tracker.values, (1, 2, 4))


def test_auto_tracker_add_multiple() -> None:
    tracker = AutoTracker()
    tracker.add("meow")
    tracker.add(1)
    assert isinstance(tracker.tracker, DiscreteTracker)
    assert objects_are_equal(dict(tracker.tracker.counter), {"meow": 1, 1: 1})


def test_auto_tracker_get_statistics() -> None:
    tracker = AutoTracker()
    tracker.add(["meow", "abc", "meow", "meow"])
    assert objects_are_equal(
        tracker.get_statistics(),
        {
            "count": 4,
            "num_unique_values": 2,
            "count_meow": 3,
            "count_abc": 1,
        },
    )


def test_auto_tracker_get_statistics_empty() -> None:
    tracker = AutoTracker()
    with raises(
        EmptyTrackerError, match="Cannot compute the statistics because the tracker is empty"
    ):
        tracker.get_statistics()


def test_auto_tracker_reset() -> None:
    tracker = AutoTracker()
    tracker.add("meow")
    assert isinstance(tracker.tracker, DiscreteTracker)
    tracker.reset()
    assert tracker.tracker.count() == 0


def test_auto_tracker_reset_empty() -> None:
    tracker = AutoTracker()
    tracker.reset()
    assert tracker.tracker is None


def test_continuous_tracker_load_state_dict() -> None:
    tracker = AutoTracker()
    tracker.add(1)
    state = {
        "count": 3,
        "max_value": 4,
        "min_value": 1,
        "sum": 7.0,
        "values": (1, 2, 4),
    }
    tracker.load_state_dict(state)
    assert tracker.tracker.count() == 3
    assert objects_are_equal(tracker.state_dict(), state)


def test_continuous_tracker_load_state_dict_empty() -> None:
    tracker = AutoTracker()
    tracker.load_state_dict({})
    assert objects_are_equal(tracker.state_dict(), {})


def test_auto_tracker_state_dict() -> None:
    tracker = AutoTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(
        tracker.state_dict(),
        {
            "count": 3,
            "max_value": 4,
            "min_value": 1,
            "sum": 7.0,
            "values": (1, 2, 4),
        },
    )


def test_auto_tracker_state_dict_empty() -> None:
    assert objects_are_equal(AutoTracker().state_dict(), {})
