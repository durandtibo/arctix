from __future__ import annotations

__all__ = [
    "check_torch",
    "is_torch_available",
]

from importlib.util import find_spec


def is_torch_available() -> bool:
    r"""Indicates if the torch package is installed or not."""
    return find_spec("torch") is not None


def check_torch() -> None:
    r"""Checks if the torch package is installed.

    Raises:
        RuntimeError if the torch package is not installed.
    """
    if not is_torch_available():
        raise RuntimeError(
            "`torch` package is required but not installed. "
            "You can install `torch` package with the command:\n\n"
            "pip install torch\n"
        )
