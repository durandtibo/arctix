from __future__ import annotations

import math
from collections.abc import Sequence
from unittest.mock import Mock

from coola import objects_are_allclose, objects_are_equal
from coola.utils.imports import is_numpy_available, is_torch_available
from pytest import mark, raises

from arctix.stats import ContinuousTracker, EmptyTrackerError
from arctix.testing import numpy_available, numpy_or_torch_available, torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()

#######################################
#     Tests for ContinuousTracker     #
#######################################


def test_continuous_tracker_str() -> None:
    assert str(ContinuousTracker()).startswith("ContinuousTracker(")


def test_continuous_tracker_add_scalar_float() -> None:
    tracker = ContinuousTracker()
    tracker.add(3.2)
    assert objects_are_equal(tracker.values, (3.2,))


def test_continuous_tracker_add_scalar_int() -> None:
    tracker = ContinuousTracker()
    tracker.add(3)
    assert objects_are_equal(tracker.values, (3,))


def test_continuous_tracker_add_scalar_float_and_int() -> None:
    tracker = ContinuousTracker()
    tracker.add(3.2)
    tracker.add(1)
    tracker.add(2.0)
    assert objects_are_equal(tracker.values, (3.2, 1, 2.0))


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([1.0, 1.1, 4.2, 2.0], dtype=torch.float),
        torch.tensor([[1.0, 1.1, 4.2, 2.0]], dtype=torch.float),
        torch.tensor([[1.0, 1.1], [4.2, 2.0]], dtype=torch.float),
        torch.tensor([[1.0], [1.1], [4.2], [2.0]], dtype=torch.float),
    ),
)
def test_continuous_tracker_add_tensor_float(tensor: torch.Tensor) -> None:
    tracker = ContinuousTracker()
    tracker.add(tensor)
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0))


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.tensor([1, 2, 4, 8], dtype=torch.long),
        torch.tensor([[1, 2, 4, 8]], dtype=torch.long),
        torch.tensor([[1, 2], [4, 8]], dtype=torch.long),
        torch.tensor([[1], [2], [4], [8]], dtype=torch.long),
    ),
)
def test_continuous_tracker_add_tensor_long(tensor: torch.Tensor) -> None:
    tracker = ContinuousTracker()
    tracker.add(tensor)
    assert objects_are_allclose(tracker.values, (1, 2, 4, 8))


@torch_available
def test_continuous_tracker_add_tensor_float_and_long() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1.0, 1.1, 4.2, 2.0], dtype=torch.float))
    tracker.add(torch.tensor([1, 2, 4], dtype=torch.long))
    tracker.add(torch.tensor(2.0, dtype=torch.float))
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0, 1, 2, 4, 2.0))


@torch_available
def test_continuous_tracker_add_tensor_empty() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([]))
    assert objects_are_equal(tracker.values, tuple())


@numpy_available
@mark.parametrize(
    "array",
    (
        np.array([1.0, 1.1, 4.2, 2.0], dtype=float),
        np.array([[1.0, 1.1, 4.2, 2.0]], dtype=float),
        np.array([[1.0, 1.1], [4.2, 2.0]], dtype=float),
        np.array([[1.0], [1.1], [4.2], [2.0]], dtype=float),
    ),
)
def test_continuous_tracker_add_ndarray_float(array: np.ndarray) -> None:
    tracker = ContinuousTracker()
    tracker.add(array)
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0))


@numpy_available
@mark.parametrize(
    "array",
    (
        np.array([1, 2, 4, 8], dtype=int),
        np.array([[1, 2, 4, 8]], dtype=int),
        np.array([[1, 2], [4, 8]], dtype=int),
        np.array([[1], [2], [4], [8]], dtype=int),
    ),
)
def test_continuous_tracker_add_ndarray_long(array: np.ndarray) -> None:
    tracker = ContinuousTracker()
    tracker.add(array)
    assert objects_are_allclose(tracker.values, (1, 2, 4, 8))


@numpy_available
def test_continuous_tracker_add_ndarray_float_and_long() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1.0, 1.1, 4.2, 2.0], dtype=float))
    tracker.add(np.array([1, 2, 4], dtype=int))
    tracker.add(np.array(2.0, dtype=float))
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0, 1, 2, 4, 2.0))


@numpy_available
def test_continuous_tracker_add_ndarray_empty() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([]))
    assert objects_are_equal(tracker.values, tuple())


@numpy_available
@mark.parametrize(
    "sequence",
    ([1.0, 1.1, 4.2, 2.0], (1.0, 1.1, 4.2, 2.0)),
)
def test_continuous_tracker_add_sequence_float(sequence: Sequence[int | float]) -> None:
    tracker = ContinuousTracker()
    tracker.add(sequence)
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0))


@numpy_available
@mark.parametrize("sequence", ([1, 2, 4, 8], (1, 2, 4, 8)))
def test_continuous_tracker_add_sequence_long(sequence: Sequence[int | float]) -> None:
    tracker = ContinuousTracker()
    tracker.add(sequence)
    assert objects_are_allclose(tracker.values, (1, 2, 4, 8))


@numpy_available
def test_continuous_tracker_add_sequence_float_and_long() -> None:
    tracker = ContinuousTracker()
    tracker.add([1.0, 1.1, 4.2, 2.0])
    tracker.add([1, 2, 4])
    tracker.add(2.0)
    assert objects_are_allclose(tracker.values, (1.0, 1.1, 4.2, 2.0, 1, 2, 4, 2.0))


@numpy_available
def test_continuous_tracker_add_sequence_empty() -> None:
    tracker = ContinuousTracker()
    tracker.add([])
    assert objects_are_equal(tracker.values, tuple())


def test_continuous_tracker_count_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(tracker.count(), 3)


@torch_available
def test_continuous_tracker_count_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 8]))
    assert objects_are_equal(tracker.count(), 4)


@numpy_available
def test_continuous_tracker_count_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2]))
    assert objects_are_equal(tracker.count(), 2)


def test_continuous_tracker_count_empty() -> None:
    assert objects_are_equal(ContinuousTracker().count(), 0)


@numpy_or_torch_available
def test_continuous_tracker_get_statistics_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    stats = tracker.get_statistics()
    assert math.isnan(stats.pop("std"))
    assert objects_are_allclose(
        stats,
        {
            "count": 1,
            "sum": 1.0,
            "mean": 1.0,
            "median": 1,
            "max": 1,
            "min": 1,
            "quantile 0.100": 1.0,
            "quantile 0.250": 1.0,
            "quantile 0.500": 1.0,
            "quantile 0.750": 1.0,
            "quantile 0.900": 1.0,
        },
    )


@numpy_or_torch_available
def test_continuous_tracker_get_statistics_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 1, 1, 1, 1, 1, 1, 1])
    assert objects_are_allclose(
        tracker.get_statistics(),
        {
            "count": 8,
            "sum": 8.0,
            "mean": 1.0,
            "median": 1,
            "max": 1,
            "min": 1,
            "std": 0.0,
            "quantile 0.100": 1.0,
            "quantile 0.250": 1.0,
            "quantile 0.500": 1.0,
            "quantile 0.750": 1.0,
            "quantile 0.900": 1.0,
        },
    )


@torch_available
def test_continuous_tracker_get_statistics_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.arange(21))
    assert objects_are_allclose(
        tracker.get_statistics(),
        {
            "count": 21,
            "sum": 210.0,
            "mean": 10.0,
            "median": 10,
            "max": 20,
            "min": 0,
            "std": 6.204836845397949,
            "quantile 0.100": 2.0,
            "quantile 0.250": 5.0,
            "quantile 0.500": 10.0,
            "quantile 0.750": 15.0,
            "quantile 0.900": 18.0,
        },
    )


@numpy_available
def test_continuous_tracker_get_statistics_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.arange(11))
    assert objects_are_allclose(
        tracker.get_statistics(),
        {
            "count": 11,
            "sum": 55.0,
            "mean": 5.0,
            "median": 5,
            "max": 10,
            "min": 0,
            "std": 3.316624879837036,
            "quantile 0.100": 1.0,
            "quantile 0.250": 2.5,
            "quantile 0.500": 5.0,
            "quantile 0.750": 7.5,
            "quantile 0.900": 9.0,
        },
    )


@numpy_or_torch_available
def test_continuous_tracker_get_statistics_no_quantiles() -> None:
    tracker = ContinuousTracker(quantiles=[])
    tracker.add([1, 1, 1, 1, 1, 1, 1, 1])
    assert objects_are_allclose(
        tracker.get_statistics(),
        {
            "count": 8,
            "sum": 8.0,
            "mean": 1.0,
            "median": 1,
            "max": 1,
            "min": 1,
            "std": 0.0,
        },
    )


def test_continuous_tracker_get_statistics_empty() -> None:
    tracker = ContinuousTracker()
    with raises(
        EmptyTrackerError, match="Cannot compute the statistics because the tracker is empty"
    ):
        tracker.get_statistics()


def test_continuous_tracker_max_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.max(), 1)


def test_continuous_tracker_max_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(tracker.max(), 4)


@torch_available
def test_continuous_tracker_max_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 8]))
    assert objects_are_equal(tracker.max(), 8)


@numpy_available
def test_continuous_tracker_max_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2]))
    assert objects_are_equal(tracker.max(), 2)


def test_continuous_tracker_max_empty() -> None:
    tracker = ContinuousTracker()
    with raises(EmptyTrackerError, match="Cannot compute the maximum because the tracker is empty"):
        tracker.max()


def test_continuous_tracker_mean_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.mean(), 1.0)


def test_continuous_tracker_mean_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 3])
    assert objects_are_equal(tracker.mean(), 2.0)


@torch_available
def test_continuous_tracker_mean_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 9]))
    assert objects_are_equal(tracker.mean(), 4.0)


@numpy_available
def test_continuous_tracker_mean_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2]))
    assert objects_are_equal(tracker.mean(), 1.5)


def test_continuous_tracker_mean_empty() -> None:
    tracker = ContinuousTracker()
    with raises(EmptyTrackerError, match="Cannot compute the mean because the tracker is empty"):
        tracker.mean()


def test_continuous_tracker_median_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.median(), 1)


def test_continuous_tracker_median_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(tracker.median(), 2)


@torch_available
def test_continuous_tracker_median_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 8, 5]))
    assert objects_are_equal(tracker.median(), 4)


@numpy_available
def test_continuous_tracker_median_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2, 0]))
    assert objects_are_equal(tracker.median(), 1)


def test_continuous_tracker_median_empty() -> None:
    tracker = ContinuousTracker()
    with raises(EmptyTrackerError, match="Cannot compute the median because the tracker is empty"):
        tracker.median()


def test_continuous_tracker_min_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.min(), 1)


def test_continuous_tracker_min_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(tracker.min(), 1)


@torch_available
def test_continuous_tracker_min_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, -8]))
    assert objects_are_equal(tracker.min(), -8)


@numpy_available
def test_continuous_tracker_min_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([-1, 2]))
    assert objects_are_equal(tracker.min(), -1)


def test_continuous_tracker_min_empty() -> None:
    tracker = ContinuousTracker()
    with raises(EmptyTrackerError, match="Cannot compute the minimum because the tracker is empty"):
        tracker.min()


@numpy_or_torch_available
def test_continuous_tracker_quantiles_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.quantiles(), [1.0, 1.0, 1.0, 1.0, 1.0])


@numpy_or_torch_available
def test_continuous_tracker_quantiles_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 1, 1, 1, 1, 1, 1, 1])
    assert objects_are_equal(tracker.quantiles(), [1.0, 1.0, 1.0, 1.0, 1.0])


@torch_available
def test_continuous_tracker_quantiles_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.arange(21))
    assert objects_are_equal(tracker.quantiles(), [2.0, 5.0, 10.0, 15.0, 18.0])


@numpy_available
def test_continuous_tracker_quantiles_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.arange(11))
    assert objects_are_equal(tracker.quantiles(), [1.0, 2.5, 5.0, 7.5, 9.0])


@numpy_or_torch_available
def test_continuous_tracker_quantiles_quantiles_2() -> None:
    tracker = ContinuousTracker(quantiles=(0.2, 0.8))
    tracker.add(list(range(11)))
    assert objects_are_equal(tracker.quantiles(), [2.0, 8.0])


def test_continuous_tracker_quantiles_empty() -> None:
    tracker = ContinuousTracker()
    with raises(
        EmptyTrackerError, match="Cannot compute the quantiles because the tracker is empty"
    ):
        tracker.quantiles()


def test_continuous_tracker_reset() -> None:
    tracker = ContinuousTracker()
    tracker.add(1.0)
    assert tracker.count() == 1
    assert tracker.values == (1.0,)
    tracker.reset()
    assert tracker.count() == 0
    assert tracker.values == ()
    assert tracker._sum == 0
    assert tracker._max_value == -float("inf")
    assert tracker._min_value == float("inf")


def test_continuous_tracker_std_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert math.isnan(tracker.std())


def test_continuous_tracker_std_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_allclose(tracker.std(), 1.5275251865386963, atol=1e6)


@torch_available
def test_continuous_tracker_std_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 8]))
    assert objects_are_allclose(tracker.std(), 3.095695972442627, atol=1e6)


@numpy_available
def test_continuous_tracker_std_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2]))
    assert objects_are_allclose(tracker.std(), 0.7071067690849304, atol=1e6)


def test_continuous_tracker_std_empty() -> None:
    tracker = ContinuousTracker()
    with raises(
        EmptyTrackerError,
        match="Cannot compute the standard deviation because the tracker is empty",
    ):
        tracker.std()


def test_continuous_tracker_sum_scalar() -> None:
    tracker = ContinuousTracker()
    tracker.add(1)
    assert objects_are_equal(tracker.sum(), 1.0)


def test_continuous_tracker_sum_list() -> None:
    tracker = ContinuousTracker()
    tracker.add([1, 2, 4])
    assert objects_are_equal(tracker.sum(), 7.0)


@torch_available
def test_continuous_tracker_sum_tensor() -> None:
    tracker = ContinuousTracker()
    tracker.add(torch.tensor([1, 2, 4, 8]))
    assert objects_are_equal(tracker.sum(), 15.0)


@numpy_available
def test_continuous_tracker_sum_ndarray() -> None:
    tracker = ContinuousTracker()
    tracker.add(np.array([1, 2]))
    assert objects_are_equal(tracker.sum(), 3.0)


def test_continuous_tracker_sum_empty() -> None:
    tracker = ContinuousTracker()
    with raises(EmptyTrackerError, match="Cannot compute the sum because the tracker is empty"):
        tracker.sum()
