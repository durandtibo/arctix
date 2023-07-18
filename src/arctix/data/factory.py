r"""This module implements some utility functions for the data
summary."""

from __future__ import annotations

__all__ = ["setup_data_summary"]

import logging

from arctix.data.base import BaseDataSummary
from arctix.data.noop import NoOpDataSummary

logger = logging.getLogger(__name__)


def setup_data_summary(data_summary: BaseDataSummary | dict | None) -> BaseDataSummary:
    r"""Sets up a data summary object.

    The data summary module is instantiated from its configuration by
    using the ``BaseDataSummary`` factory function.

    Args:
    ----
        data_summary (``BaseDataSummary`` or dict or ``None``):
            Specifies the data summary object or its configuration.
            If ``None``, a ``NoOpDataSummary`` object is returned.

    Returns:
    -------
        ``BaseDataSummary``: The data summary object.
    """
    if data_summary is None:
        return NoOpDataSummary()
    if isinstance(data_summary, dict):
        logger.info("Initializing the data summary module from its configuration... ")
        data_summary = BaseDataSummary.factory(**data_summary)
    return data_summary
