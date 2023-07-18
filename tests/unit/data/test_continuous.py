from __future__ import annotations

import math
from unittest.mock import Mock

from coola import objects_are_allclose
from pytest import mark, raises

from arctix import is_torch_available
from arctix.data import EmptyDataSummaryError
from arctix.data.continuous import (
    FloatDataSummary,
    FloatTensorDataSummary,
    FloatTensorSequenceDataSummary,
    prepare_quantiles,
)
from arctix.testing import torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover

#######################################
#     Tests for prepare_quantiles     #
#######################################


@torch_available
@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8)))
def test_prepare_quantiles_list_or_tuple(quantiles: list[float] | tuple[float, ...]) -> None:
    assert prepare_quantiles(quantiles).equal(torch.tensor([0.2, 0.8], dtype=torch.float))


@torch_available
def test_prepare_quantiles_torch_tensor() -> None:
    assert prepare_quantiles(torch.tensor([0.2, 0.8])).equal(
        torch.tensor([0.2, 0.8], dtype=torch.float)
    )


@torch_available
def test_prepare_quantiles_torch_tensor_sort_values() -> None:
    assert prepare_quantiles(torch.tensor([0.8, 0.2])).equal(
        torch.tensor([0.2, 0.8], dtype=torch.float)
    )


######################################
#     Tests for FloatDataSummary     #
######################################


@torch_available
def test_float_data_summary_str() -> None:
    assert str(FloatDataSummary()).startswith("FloatDataSummary(")


@torch_available
def test_float_data_summary_add_one_call() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    assert tuple(summary._values) == (0.0,)


@torch_available
def test_float_data_summary_add_two_calls() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert tuple(summary._values) == (3.0, 1.0)


@torch_available
def test_float_data_summary_count_1() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    assert summary.count() == 1


@torch_available
def test_float_data_summary_count_2() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    summary.add(0.0)
    assert summary.count() == 2


@torch_available
def test_float_data_summary_count_empty() -> None:
    summary = FloatDataSummary()
    assert summary.count() == 0


@torch_available
def test_float_data_summary_max() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.max() == 3.0


@torch_available
def test_float_data_summary_max_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.max()


@torch_available
def test_float_data_summary_mean() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.mean() == 2.0


@torch_available
def test_float_data_summary_mean_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.mean()


@torch_available
def test_float_data_summary_median() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.median() == 1.0


@torch_available
def test_float_data_summary_median_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.median()


@torch_available
def test_float_data_summary_min() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.min() == 1.0


@torch_available
def test_float_data_summary_min_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.min()


@torch_available
def test_float_data_summary_quantiles_default_quantiles() -> None:
    summary = FloatDataSummary()
    for i in range(21):
        summary.add(i)
    assert summary.quantiles().equal(
        torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    )


@torch_available
@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8), torch.tensor([0.2, 0.8]), [0.8, 0.2]))
def test_float_data_summary_quantiles_custom_quantiles(
    quantiles: torch.Tensor | tuple[float, ...] | list[float]
) -> None:
    summary = FloatDataSummary(quantiles=quantiles)
    for i in range(6):
        summary.add(i)
    assert summary.quantiles().equal(torch.tensor([1.0, 4.0]))


@torch_available
def test_float_data_summary_quantiles_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.quantiles()


@torch_available
def test_float_data_summary_std_1_value() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    assert math.isnan(summary.std())


@torch_available
def test_float_data_summary_std_0() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.add(1.0)
    assert summary.std() == 0.0


@torch_available
def test_float_data_summary_std_2() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.add(-1.0)
    assert math.isclose(summary.std(), math.sqrt(2), abs_tol=1e-6)


@torch_available
def test_float_data_summary_std_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.std()


@torch_available
def test_float_data_summary_sum() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.sum() == 4.0


@torch_available
def test_float_data_summary_sum_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.sum()


@torch_available
def test_float_data_summary_values_empty() -> None:
    summary = FloatDataSummary()
    assert tuple(summary._values) == ()


@torch_available
def test_float_data_summary_reset() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.reset()
    assert tuple(summary._values) == ()


@torch_available
def test_float_data_summary_summary_default_quantiles() -> None:
    summary = FloatDataSummary()
    for i in range(21):
        summary.add(i)
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 21,
            "sum": 210.0,
            "mean": 10.0,
            "median": 10.0,
            "std": 6.204836845397949,
            "max": 20.0,
            "min": 0.0,
            "quantile 0.000": 0.0,
            "quantile 0.100": 2.0,
            "quantile 0.200": 4.0,
            "quantile 0.300": 6.0,
            "quantile 0.400": 8.0,
            "quantile 0.500": 10.0,
            "quantile 0.600": 12.0,
            "quantile 0.700": 14.0,
            "quantile 0.800": 16.0,
            "quantile 0.900": 18.0,
            "quantile 1.000": 20.0,
        },
    )


@torch_available
@mark.parametrize("quantiles", ([], (), torch.tensor([])))
def test_float_data_summary_summary_default_no_quantile(
    quantiles: torch.Tensor | tuple[float, ...] | list[float]
) -> None:
    summary = FloatDataSummary(quantiles=quantiles)
    for _ in range(5):
        summary.add(1.0)
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 5,
            "sum": 5.0,
            "mean": 1.0,
            "median": 1.0,
            "std": 0.0,
            "max": 1.0,
            "min": 1.0,
        },
    )


@torch_available
def test_float_data_summary_summary_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()


############################################
#     Tests for FloatTensorDataSummary     #
############################################


@torch_available
def test_float_tensor_data_summary_str() -> None:
    assert str(FloatTensorDataSummary()).startswith("FloatTensorDataSummary(")


@torch_available
def test_float_tensor_data_summary_add_1_tensor() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (0.0, 3.0, 1.0, 4.0)


@torch_available
def test_float_tensor_data_summary_add_2_tensors() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor(0, dtype=torch.float))
    summary.add(torch.tensor([3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (0.0, 3.0, 1.0, 4.0)


@torch_available
@mark.parametrize(
    "tensor",
    (
        torch.ones(3 * 4 * 5),
        torch.ones(3, 4 * 5),
        torch.ones(3 * 4, 5),
        torch.ones(3, 4, 5),
        torch.ones(3, 4, 5, dtype=torch.int),
        torch.ones(3, 4, 5, dtype=torch.long),
        torch.ones(3, 4, 5, dtype=torch.double),
    ),
)
def test_float_tensor_data_summary_add_tensor(tensor: torch.Tensor) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert tuple(summary._values) == (1.0,) * 60


@torch_available
def test_float_tensor_data_summary_add_max_size_3() -> None:
    summary = FloatTensorDataSummary(max_size=3)
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (3.0, 1.0, 4.0)


@torch_available
def test_float_tensor_data_summary_add_empty_tensor() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([]))
    assert tuple(summary._values) == ()


@torch_available
@mark.parametrize(
    "tensor,count", ((torch.ones(5), 5), (torch.tensor([4, 2]), 2), (torch.arange(11), 11))
)
def test_float_tensor_data_summary_count(tensor: torch.Tensor, count: int) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.count() == count


@torch_available
def test_float_tensor_data_summary_count_empty() -> None:
    summary = FloatTensorDataSummary()
    assert summary.count() == 0


@torch_available
@mark.parametrize(
    "tensor,max_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 4.0), (torch.arange(11), 10.0)),
)
def test_float_tensor_data_summary_max(tensor: torch.Tensor, max_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.max() == max_value


@torch_available
def test_float_tensor_data_summary_max_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.max()


@torch_available
@mark.parametrize(
    "tensor,mean_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 3.0), (torch.arange(11), 5.0)),
)
def test_float_tensor_data_summary_mean(tensor: torch.Tensor, mean_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.mean() == mean_value


@torch_available
def test_float_tensor_data_summary_mean_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.mean()


@torch_available
@mark.parametrize(
    "tensor,median_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 2.0), (torch.arange(11), 5.0)),
)
def test_float_tensor_data_summary_median(tensor: torch.Tensor, median_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.median() == median_value


@torch_available
def test_float_tensor_data_summary_median_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.median()


@torch_available
@mark.parametrize(
    "tensor,min_value", ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 2.0), (torch.arange(11), 0.0))
)
def test_float_tensor_data_summary_min(tensor: torch.Tensor, min_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.min() == min_value


@torch_available
def test_float_tensor_data_summary_min_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.min()


@torch_available
def test_float_tensor_data_summary_quantiles_default_quantiles() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.arange(21))
    assert summary.quantiles().equal(
        torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    )


@torch_available
@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8), torch.tensor([0.2, 0.8]), [0.8, 0.2]))
def test_float_tensor_data_summary_quantiles_custom_quantiles(
    quantiles: torch.Tensor | tuple[float, ...] | list[float]
) -> None:
    summary = FloatTensorDataSummary(quantiles=quantiles)
    summary.add(torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float))
    assert summary.quantiles().equal(torch.tensor([1.0, 4.0]))


@torch_available
def test_float_tensor_data_summary_quantiles_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.quantiles()


@torch_available
@mark.parametrize("tensor,std_value", ((torch.ones(5), 0.0), (torch.tensor([-1, 1]), math.sqrt(2))))
def test_float_tensor_data_summary_std(tensor: torch.Tensor, std_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert math.isclose(summary.std(), std_value, abs_tol=1e-6)


@torch_available
def test_float_tensor_data_summary_std_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.std()


@torch_available
@mark.parametrize(
    "tensor,sum_value",
    ((torch.ones(5), 5.0), (torch.tensor([4, 2]), 6.0), (torch.arange(11), 55.0)),
)
def test_float_tensor_data_summary_sum(tensor: torch.Tensor, sum_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.sum() == sum_value


@torch_available
def test_float_tensor_data_summary_sum_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.sum()


@torch_available
def test_float_tensor_data_summary_values_empty() -> None:
    summary = FloatTensorDataSummary()
    assert tuple(summary._values) == ()


@torch_available
def test_float_tensor_data_summary_reset() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    summary.reset()
    assert tuple(summary._values) == ()


@torch_available
def test_float_tensor_data_summary_summary_default_quantiles() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.arange(21))
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 21,
            "sum": 210.0,
            "mean": 10.0,
            "median": 10.0,
            "std": 6.204836845397949,
            "max": 20.0,
            "min": 0.0,
            "quantile 0.000": 0.0,
            "quantile 0.100": 2.0,
            "quantile 0.200": 4.0,
            "quantile 0.300": 6.0,
            "quantile 0.400": 8.0,
            "quantile 0.500": 10.0,
            "quantile 0.600": 12.0,
            "quantile 0.700": 14.0,
            "quantile 0.800": 16.0,
            "quantile 0.900": 18.0,
            "quantile 1.000": 20.0,
        },
    )


@torch_available
@mark.parametrize("quantiles", ([], (), torch.tensor([])))
def test_float_tensor_data_summary_summary_default_no_quantile(
    quantiles: torch.Tensor | tuple[float, ...] | list[float]
) -> None:
    summary = FloatTensorDataSummary(quantiles=quantiles)
    summary.add(torch.ones(5))
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 5,
            "sum": 5.0,
            "mean": 1.0,
            "median": 1.0,
            "std": 0.0,
            "max": 1.0,
            "min": 1.0,
        },
    )


@torch_available
def test_float_tensor_data_summary_summary_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()


####################################################
#     Tests for FloatTensorSequenceDataSummary     #
####################################################


@torch_available
def test_float_tensor_sequence_data_summary_str() -> None:
    assert str(FloatTensorSequenceDataSummary()).startswith("FloatTensorSequenceDataSummary(")


@torch_available
def test_float_tensor_sequence_data_summary_add() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert summary._value.count() == 4
    assert summary._length.count() == 1


@torch_available
def test_float_tensor_sequence_data_summary_reset() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    summary.reset()
    assert summary._value.count() == 0
    assert summary._length.count() == 0


@torch_available
def test_float_tensor_sequence_data_summary_summary() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    stats = summary.summary()
    assert len(stats) == 2
    assert "value" in stats
    assert "length" in stats


@torch_available
def test_float_tensor_sequence_data_summary_summary_empty() -> None:
    summary = FloatTensorSequenceDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()
