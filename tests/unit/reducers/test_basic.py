from __future__ import annotations

import math
from collections.abc import Sequence

from pytest import mark, raises

from arctix.reducers import BasicReducer, EmptySequenceError, ReducerRegistry


def test_reducer_registry_available_reducers() -> None:
    assert isinstance(ReducerRegistry.registry["basic"], BasicReducer)


##################################
#     Tests for BasicReducer     #
##################################


def test_basic_reducer_str() -> None:
    assert str(BasicReducer()).startswith("BasicReducer(")


@mark.parametrize("values", ([-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3]))
def test_basic_reducer_max_int(values: Sequence[int | float]) -> None:
    val = BasicReducer().max(values)
    assert isinstance(val, int)
    assert val == 2


@mark.parametrize(
    "values", ([-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [2.5], [2.5, -1.5, -2, -3])
)
def test_basic_reducer_max_float(values: Sequence[int | float]) -> None:
    val = BasicReducer().max(values)
    assert isinstance(val, float)
    assert val == 2.5


@mark.parametrize("values", ([], ()))
def test_basic_reducer_max_empty(values: Sequence[int | float]) -> None:
    with raises(
        EmptySequenceError, match="Cannot compute the maximum because the summary is empty"
    ):
        BasicReducer().max([])


@mark.parametrize("values", ([-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]))
def test_basic_reducer_mean_int(values: Sequence[int | float]) -> None:
    val = BasicReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.0


@mark.parametrize("values", ([-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]))
def test_basic_reducer_mean_float(values: Sequence[int | float]) -> None:
    val = BasicReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.5


@mark.parametrize("values", ([], ()))
def test_basic_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with raises(EmptySequenceError, match="Cannot compute the mean because the summary is empty"):
        BasicReducer().mean([])


@mark.parametrize("values", ([-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]))
def test_basic_reducer_median_int(values: Sequence[int | float]) -> None:
    val = BasicReducer().median(values)
    assert isinstance(val, int)
    assert val == 0


@mark.parametrize("values", ([-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]))
def test_basic_reducer_median_float(values: Sequence[int | float]) -> None:
    val = BasicReducer().median(values)
    assert isinstance(val, float)
    assert val == 0.5


@mark.parametrize("values", ([], ()))
def test_basic_reducer_median_empty(values: Sequence[int | float]) -> None:
    with raises(EmptySequenceError, match="Cannot compute the median because the summary is empty"):
        BasicReducer().median([])


@mark.parametrize("values", ([-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3]))
def test_basic_reducer_min_int(values: Sequence[int | float]) -> None:
    val = BasicReducer().min(values)
    assert isinstance(val, int)
    assert val == -2


@mark.parametrize(
    "values", ([-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [-2.5], [-2.5, 1.5, 2, 3])
)
def test_basic_reducer_min_float(values: Sequence[int | float]) -> None:
    val = BasicReducer().min(values)
    assert isinstance(val, float)
    assert val == -2.5


@mark.parametrize("values", ([], ()))
def test_basic_reducer_min_empty(values: Sequence[int | float]) -> None:
    with raises(
        EmptySequenceError, match="Cannot compute the minimum because the summary is empty"
    ):
        BasicReducer().min([])


@mark.parametrize(
    "values", ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
)
def test_basic_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert BasicReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@mark.parametrize(
    "values",
    (
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
    ),
)
def test_basic_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert BasicReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@mark.parametrize("values", ([], ()))
def test_basic_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with raises(
        EmptySequenceError, match="Cannot compute the quantiles because the summary is empty"
    ):
        BasicReducer().quantile([], [0.5])


@mark.parametrize("values", ([2, 1, -2, 3, 0], (2, 1, -2, 3, 0)))
def test_basic_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert BasicReducer().sort(values) == [-2, 0, 1, 2, 3]


@mark.parametrize("values", ([2.5, 1.5, -2.5, 3.5, 0.5], (2.5, 1.5, -2.5, 3.5, 0.5)))
def test_basic_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert BasicReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@mark.parametrize("values", ([2, 1, -2, 3, 0], (2, 1, -2, 3, 0)))
def test_basic_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert BasicReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@mark.parametrize("values", ([], ()))
def test_basic_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert BasicReducer().sort([]) == []


@mark.parametrize("values", ([-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2)))
def test_basic_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(BasicReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@mark.parametrize("values", ([-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5)))
def test_basic_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(BasicReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@mark.parametrize("values", ([1], [1.0]))
def test_basic_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(BasicReducer().std(values))


@mark.parametrize("values", ([], ()))
def test_basic_reducer_std_empty(values: Sequence[int | float]) -> None:
    with raises(
        EmptySequenceError,
        match="Cannot compute the standard deviation because the summary is empty",
    ):
        BasicReducer().std([])
