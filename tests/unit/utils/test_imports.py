from __future__ import annotations

from unittest.mock import patch

import pytest

from arctix.utils.imports import check_gdown, gdown_available, is_gdown_available


def my_function(n: int = 0) -> int:
    return 42 + n


#################
#     gdown     #
#################


def test_check_gdown_with_package() -> None:
    with patch("arctix.utils.imports.is_gdown_available", lambda: True):
        check_gdown()


def test_check_gdown_without_package() -> None:
    with (
        patch("arctix.utils.imports.is_gdown_available", lambda: False),
        pytest.raises(RuntimeError, match="`gdown` package is required but not installed."),
    ):
        check_gdown()


def test_is_gdown_available() -> None:
    assert isinstance(is_gdown_available(), bool)


def test_gdown_available_with_package() -> None:
    with patch("arctix.utils.imports.is_gdown_available", lambda: True):
        fn = gdown_available(my_function)
        assert fn(2) == 44


def test_gdown_available_without_package() -> None:
    with patch("arctix.utils.imports.is_gdown_available", lambda: False):
        fn = gdown_available(my_function)
        assert fn(2) is None


def test_gdown_available_decorator_with_package() -> None:
    with patch("arctix.utils.imports.is_gdown_available", lambda: True):

        @gdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_gdown_available_decorator_without_package() -> None:
    with patch("arctix.utils.imports.is_gdown_available", lambda: False):

        @gdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None