r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_gdown",
    "is_gdown_available",
    "gdown_available",
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
