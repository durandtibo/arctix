from __future__ import annotations

from unittest.mock import patch

from pytest import raises

from arctix.utils.imports import (
    check_numpy,
    check_torch,
    is_numpy_available,
    is_torch_available,
)


def test_check_numpy_with_package() -> None:
    with patch("arctix.utils.imports.is_numpy_available", lambda *args: True):
        check_numpy()


def test_check_numpy_without_package() -> None:
    with patch("arctix.utils.imports.is_numpy_available", lambda *args: False):
        with raises(RuntimeError, match="`numpy` package is required but not installed."):
            check_numpy()


def test_is_numpy_available() -> None:
    assert isinstance(is_numpy_available(), bool)


def test_check_torch_with_package() -> None:
    with patch("arctix.utils.imports.is_torch_available", lambda *args: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with patch("arctix.utils.imports.is_torch_available", lambda *args: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)
