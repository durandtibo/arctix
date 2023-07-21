from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import Mock

from coola import objects_are_allclose, objects_are_equal
from coola.utils.imports import is_numpy_available, is_torch_available
from pytest import mark, raises

from arctix.stats import DiscreteTracker, EmptyTrackerError
from arctix.testing import numpy_available, torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()

#####################################
#     Tests for DiscreteTracker     #
#####################################


def test_discrete_tracker_str() -> None:
    assert str(DiscreteTracker()).startswith("DiscreteTracker(")


@mark.parametrize("data", (True, False, 1, 42, 4.2, 2.0, "meow", "abc"))
def test_discrete_tracker_add_scalar(data: bool | float | int | str) -> None:
    tracker = DiscreteTracker()
    tracker.add(data)
    assert objects_are_equal(dict(tracker.counter), {data: 1})


@mark.parametrize(
    "sequence",
    (
        [True, False, True, True],
        (True, False, True, True),
        [True, True, True, False],
    ),
)
def test_discrete_tracker_add_sequence_bool(sequence: Sequence[bool]) -> None:
    tracker = DiscreteTracker()
    tracker.add(sequence)
    assert objects_are_equal(dict(tracker.counter), {True: 3, False: 1})


@mark.parametrize(
    "sequence",
    (
        [1.0, 1.1, 4.2, 1.0],
        (1.0, 1.1, 4.2, 1.0),
        [1.0, 1.0, 1.1, 4.2],
    ),
)
def test_discrete_tracker_add_sequence_float(sequence: Sequence[bool | float | int | str]) -> None:
    tracker = DiscreteTracker()
    tracker.add(sequence)
    assert objects_are_equal(dict(tracker.counter), {1.0: 2, 1.1: 1, 4.2: 1})


@mark.parametrize(
    "sequence",
    (
        [1, 2, 4, 1],
        (1, 2, 4, 1),
        [1, 1, 2, 4],
    ),
)
def test_discrete_tracker_add_sequence_int(sequence: Sequence[bool | float | int | str]) -> None:
    tracker = DiscreteTracker()
    tracker.add(sequence)
    assert objects_are_equal(dict(tracker.counter), {1: 2, 2: 1, 4: 1})


@mark.parametrize(
    "sequence",
    (
        ["meow", "abc", "meow", "meow"],
        ("meow", "abc", "meow", "meow"),
        ["meow", "meow", "meow", "abc"],
    ),
)
def test_discrete_tracker_add_sequence_str(sequence: Sequence[str]) -> None:
    tracker = DiscreteTracker()
    tracker.add(sequence)
    assert objects_are_equal(dict(tracker.counter), {"meow": 3, "abc": 1})


@numpy_available
@mark.parametrize(
    "array",
    (
        np.array([1.0, 1.1, 4.2, 1.0], dtype=float),
        np.array([[1.0, 1.1, 4.2, 1.0]], dtype=float),
        np.array([[1.0, 1.1], [4.2, 1.0]], dtype=float),
        np.array([[1.0], [1.1], [4.2], [1.0]], dtype=float),
        np.array([1.0, 1.0, 1.1, 4.2], dtype=float),
    ),
)
def test_discrete_tracker_add_ndarray_float(array: np.ndarray) -> None:
    tracker = DiscreteTracker()
    tracker.add(array)
    assert objects_are_equal(dict(tracker.counter), {1.0: 2, 1.1: 1, 4.2: 1})


@numpy_available
@mark.parametrize(
    "array",
    (
        np.array([1, 2, 4, 1], dtype=int),
        np.array([[1, 2, 4, 1]], dtype=int),
        np.array([[1, 2], [4, 1]], dtype=int),
        np.array([[1], [2], [4], [1]], dtype=int),
        np.array([1, 1, 2, 4], dtype=int),
    ),
)
def test_discrete_tracker_add_ndarray_int(array: np.ndarray) -> None:
    tracker = DiscreteTracker()
    tracker.add(array)
    assert objects_are_equal(dict(tracker.counter), {1: 2, 2: 1, 4: 1})


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([1.0, 1.1, 4.2, 1.0], dtype=torch.float),
        torch.tensor([[1.0, 1.1, 4.2, 1.0]], dtype=torch.float),
        torch.tensor([[1.0, 1.1], [4.2, 1.0]], dtype=torch.float),
        torch.tensor([[1.0], [1.1], [4.2], [1.0]], dtype=torch.float),
        torch.tensor([1.0, 1.0, 1.1, 4.2], dtype=torch.float),
        torch.tensor([1.0, 1.1, 4.2, 1.0], dtype=torch.double),
    ),
)
def test_discrete_tracker_add_tensor_float(tensor: torch.Tensor) -> None:
    tracker = DiscreteTracker()
    tracker.add(tensor)
    assert objects_are_allclose(dict(tracker.counter), {1.0: 2, 1.1: 1, 4.2: 1}, atol=1e-6)


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([1, 2, 4, 1], dtype=torch.long),
        torch.tensor([[1, 2, 4, 1]], dtype=torch.long),
        torch.tensor([[1, 2], [4, 1]], dtype=torch.long),
        torch.tensor([[1], [2], [4], [1]], dtype=torch.long),
        torch.tensor([1, 1, 2, 4], dtype=torch.long),
        torch.tensor([1, 2, 4, 1], dtype=torch.int),
    ),
)
def test_discrete_tracker_add_tensor_int(tensor: torch.Tensor) -> None:
    tracker = DiscreteTracker()
    tracker.add(tensor)
    assert objects_are_equal(dict(tracker.counter), {1: 2, 2: 1, 4: 1})


@mark.parametrize("data", (True, 1, 4.2, "meow"))
def test_discrete_tracker_count_1(data: bool | float | int | str) -> None:
    tracker = DiscreteTracker()
    tracker.add(data)
    assert tracker.count() == 1


def test_discrete_tracker_count_2() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 1])
    assert tracker.count() == 2


def test_discrete_tracker_count_empty() -> None:
    assert DiscreteTracker().count() == 0


def test_discrete_tracker_get_statistics() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 2, 4, 1])
    assert objects_are_equal(
        tracker.get_statistics(),
        {
            "count": 4,
            "num_unique_values": 3,
            "count_1": 2,
            "count_2": 1,
            "count_4": 1,
        },
    )


def test_discrete_tracker_get_statistics_empty() -> None:
    tracker = DiscreteTracker()
    with raises(
        EmptyTrackerError, match="Cannot compute the statistics because the tracker is empty"
    ):
        tracker.get_statistics()


def test_discrete_tracker_most_common() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 1, 2, 3, 1, 2, 2, 1])
    assert tracker.most_common() == [(1, 4), (2, 3), (3, 1)]


def test_discrete_tracker_most_common_2() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 1, 2, 3, 1, 2, 2, 1])
    assert tracker.most_common(2) == [(1, 4), (2, 3)]


def test_discrete_tracker_most_common_5() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 1, 2, 3, 1, 2, 2, 1])
    assert tracker.most_common(5) == [(1, 4), (2, 3), (3, 1)]


def test_discrete_tracker_most_common_empty() -> None:
    tracker = DiscreteTracker()
    with raises(
        EmptyTrackerError,
        match="Cannot compute the most common values because the tracker is empty",
    ):
        tracker.most_common()


def test_discrete_tracker_reset() -> None:
    tracker = DiscreteTracker()
    tracker.add([1, 1])
    assert objects_are_equal(dict(tracker.counter), {1: 2})
    assert tracker.count() == 2
    tracker.reset()
    assert objects_are_equal(dict(tracker.counter), {})
    assert tracker.count() == 0


def test_discrete_tracker_reset_empty() -> None:
    tracker = DiscreteTracker()
    tracker.reset()
    assert objects_are_equal(dict(tracker.counter), {})
    assert tracker.count() == 0
