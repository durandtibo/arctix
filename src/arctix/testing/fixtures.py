r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = ["gdown_available", "requests_available", "tqdm_available"]

import pytest

from arctix.utils.imports import (
    is_gdown_available,
    is_requests_available,
    is_tqdm_available,
)

gdown_available = pytest.mark.skipif(not is_gdown_available(), reason="Require gdown")
requests_available = pytest.mark.skipif(not is_requests_available(), reason="Require requests")
tqdm_available = pytest.mark.skipif(not is_tqdm_available(), reason="Require tqdm")
