r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = ["gdown_available"]

import pytest

from arctix.utils.imports import is_gdown_available

gdown_available = pytest.mark.skipif(not is_gdown_available(), reason="Require gdown")
