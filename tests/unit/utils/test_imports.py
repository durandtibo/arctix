from __future__ import annotations

from unittest.mock import patch

import pytest

from arctix.utils.imports import (
    check_gdown,
    check_matplotlib,
    check_requests,
    check_tqdm,
    gdown_available,
    is_gdown_available,
    is_matplotlib_available,
    is_requests_available,
    is_tqdm_available,
    matplotlib_available,
    requests_available,
    tqdm_available,
)


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


######################
#     matplotlib     #
######################


def test_check_matplotlib_with_package() -> None:
    with patch("arctix.utils.imports.is_matplotlib_available", lambda: True):
        check_matplotlib()


def test_check_matplotlib_without_package() -> None:
    with (
        patch("arctix.utils.imports.is_matplotlib_available", lambda: False),
        pytest.raises(RuntimeError, match="`matplotlib` package is required but not installed."),
    ):
        check_matplotlib()


def test_is_matplotlib_available() -> None:
    assert isinstance(is_matplotlib_available(), bool)


def test_matplotlib_available_with_package() -> None:
    with patch("arctix.utils.imports.is_matplotlib_available", lambda: True):
        fn = matplotlib_available(my_function)
        assert fn(2) == 44


def test_matplotlib_available_without_package() -> None:
    with patch("arctix.utils.imports.is_matplotlib_available", lambda: False):
        fn = matplotlib_available(my_function)
        assert fn(2) is None


def test_matplotlib_available_decorator_with_package() -> None:
    with patch("arctix.utils.imports.is_matplotlib_available", lambda: True):

        @matplotlib_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_matplotlib_available_decorator_without_package() -> None:
    with patch("arctix.utils.imports.is_matplotlib_available", lambda: False):

        @matplotlib_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


####################
#     requests     #
####################


def test_check_requests_with_package() -> None:
    with patch("arctix.utils.imports.is_requests_available", lambda: True):
        check_requests()


def test_check_requests_without_package() -> None:
    with (
        patch("arctix.utils.imports.is_requests_available", lambda: False),
        pytest.raises(RuntimeError, match="`requests` package is required but not installed."),
    ):
        check_requests()


def test_is_requests_available() -> None:
    assert isinstance(is_requests_available(), bool)


def test_requests_available_with_package() -> None:
    with patch("arctix.utils.imports.is_requests_available", lambda: True):
        fn = requests_available(my_function)
        assert fn(2) == 44


def test_requests_available_without_package() -> None:
    with patch("arctix.utils.imports.is_requests_available", lambda: False):
        fn = requests_available(my_function)
        assert fn(2) is None


def test_requests_available_decorator_with_package() -> None:
    with patch("arctix.utils.imports.is_requests_available", lambda: True):

        @requests_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_requests_available_decorator_without_package() -> None:
    with patch("arctix.utils.imports.is_requests_available", lambda: False):

        @requests_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


################
#     tqdm     #
################


def test_check_tqdm_with_package() -> None:
    with patch("arctix.utils.imports.is_tqdm_available", lambda: True):
        check_tqdm()


def test_check_tqdm_without_package() -> None:
    with (
        patch("arctix.utils.imports.is_tqdm_available", lambda: False),
        pytest.raises(RuntimeError, match="`tqdm` package is required but not installed."),
    ):
        check_tqdm()


def test_is_tqdm_available() -> None:
    assert isinstance(is_tqdm_available(), bool)


def test_tqdm_available_with_package() -> None:
    with patch("arctix.utils.imports.is_tqdm_available", lambda: True):
        fn = tqdm_available(my_function)
        assert fn(2) == 44


def test_tqdm_available_without_package() -> None:
    with patch("arctix.utils.imports.is_tqdm_available", lambda: False):
        fn = tqdm_available(my_function)
        assert fn(2) is None


def test_tqdm_available_decorator_with_package() -> None:
    with patch("arctix.utils.imports.is_tqdm_available", lambda: True):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_tqdm_available_decorator_without_package() -> None:
    with patch("arctix.utils.imports.is_tqdm_available", lambda: False):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
