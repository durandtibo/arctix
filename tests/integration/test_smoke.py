from __future__ import annotations

from arctix.utils.imports import is_gdown_available


def test_is_gdown_available() -> None:
    assert isinstance(is_gdown_available(), bool)
