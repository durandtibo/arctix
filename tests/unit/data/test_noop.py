from typing import Any

from pytest import mark

from arctix.data import NoOpDataSummary

#####################################
#     Tests for NoOpDataSummary     #
#####################################


def test_noop_data_summary_str() -> None:
    assert str(NoOpDataSummary()).startswith("NoOpDataSummary(")


@mark.parametrize("data", (1, "abc"))
def test_noop_data_summary_add(data: Any) -> None:
    summary = NoOpDataSummary()
    summary.add(data)  # check it does not raise error


def test_noop_data_summary_reset() -> None:
    summary = NoOpDataSummary()
    summary.add(1)
    summary.reset()  # check it does not raise error


def test_noop_data_summary_summary() -> None:
    summary = NoOpDataSummary()
    assert summary.summary() == {}
