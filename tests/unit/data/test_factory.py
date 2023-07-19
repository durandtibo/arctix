from objectory import OBJECT_TARGET

from arctix.data import NoOpDataSummary, setup_data_summary

########################################
#     Tests for setup_data_summary     #
########################################


def test_setup_data_summary_object() -> None:
    summary = NoOpDataSummary()
    assert setup_data_summary(summary) is summary


def test_setup_data_summary_dict() -> None:
    assert isinstance(
        setup_data_summary({OBJECT_TARGET: "arctix.data.NoOpDataSummary"}),
        NoOpDataSummary,
    )


def test_setup_data_summary_none() -> None:
    assert isinstance(setup_data_summary(None), NoOpDataSummary)
