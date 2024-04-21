r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_gdown",
    "check_matplotlib",
    "check_requests",
    "check_tqdm",
    "gdown_available",
    "is_gdown_available",
    "is_matplotlib_available",
    "is_requests_available",
    "is_tqdm_available",
    "matplotlib_available",
    "requests_available",
    "tqdm_available",
]

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available

if TYPE_CHECKING:
    from collections.abc import Callable

#################
#     gdown     #
#################


def is_gdown_available() -> bool:
    r"""Indicate if the ``gdown`` package is installed or not.

    Returns:
        ``True`` if ``gdown`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import is_gdown_available
    >>> is_gdown_available()

    ```
    """
    return find_spec("gdown") is not None


def check_gdown() -> None:
    r"""Check if the ``gdown`` package is installed.

    Raises:
        RuntimeError: if the ``gdown`` package is not installed.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import check_gdown
    >>> check_gdown()

    ```
    """
    if not is_gdown_available():
        msg = (
            "`gdown` package is required but not installed. "
            "You can install `gdown` package with the command:\n\n"
            "pip install gdown\n"
        )
        raise RuntimeError(msg)


def gdown_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``gdown``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``gdown`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import gdown_available
    >>> @gdown_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_gdown_available)


######################
#     matplotlib     #
######################


def is_matplotlib_available() -> bool:
    r"""Indicate if the ``matplotlib`` package is installed or not.

    Returns:
        ``True`` if ``matplotlib`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import is_matplotlib_available
    >>> is_matplotlib_available()

    ```
    """
    return find_spec("matplotlib") is not None


def check_matplotlib() -> None:
    r"""Check if the ``matplotlib`` package is installed.

    Raises:
        RuntimeError: if the ``matplotlib`` package is not installed.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import check_matplotlib
    >>> check_matplotlib()

    ```
    """
    if not is_matplotlib_available():
        msg = (
            "`matplotlib` package is required but not installed. "
            "You can install `matplotlib` package with the command:\n\n"
            "pip install matplotlib\n"
        )
        raise RuntimeError(msg)


def matplotlib_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if
    ``matplotlib`` package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``matplotlib`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import matplotlib_available
    >>> @matplotlib_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_matplotlib_available)


####################
#     requests     #
####################


def is_requests_available() -> bool:
    r"""Indicate if the ``requests`` package is installed or not.

    Returns:
        ``True`` if ``requests`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import is_requests_available
    >>> is_requests_available()

    ```
    """
    return find_spec("requests") is not None


def check_requests() -> None:
    r"""Check if the ``requests`` package is installed.

    Raises:
        RuntimeError: if the ``requests`` package is not installed.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import check_requests
    >>> check_requests()

    ```
    """
    if not is_requests_available():
        msg = (
            "`requests` package is required but not installed. "
            "You can install `requests` package with the command:\n\n"
            "pip install requests\n"
        )
        raise RuntimeError(msg)


def requests_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``requests``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``requests`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import requests_available
    >>> @requests_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_requests_available)


################
#     tqdm     #
################


def is_tqdm_available() -> bool:
    r"""Indicate if the ``tqdm`` package is installed or not.

    Returns:
        ``True`` if ``tqdm`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import is_tqdm_available
    >>> is_tqdm_available()

    ```
    """
    return find_spec("tqdm") is not None


def check_tqdm() -> None:
    r"""Check if the ``tqdm`` package is installed.

    Raises:
        RuntimeError: if the ``tqdm`` package is not installed.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import check_tqdm
    >>> check_tqdm()

    ```
    """
    if not is_tqdm_available():
        msg = (
            "`tqdm` package is required but not installed. "
            "You can install `tqdm` package with the command:\n\n"
            "pip install tqdm\n"
        )
        raise RuntimeError(msg)


def tqdm_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``tqdm``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``tqdm`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from arctix.utils.imports import tqdm_available
    >>> @tqdm_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_tqdm_available)
