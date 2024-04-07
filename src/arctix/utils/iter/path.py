r"""Contain some iterables for paths."""

from __future__ import annotations

__all__ = ["DirFilter", "FileFilter", "PathLister"]

from collections.abc import Iterable, Iterator
from pathlib import Path

from coola.utils import str_indent, str_mapping


class DirFilter(Iterable[Path]):
    r"""Implement an iterable to keep only the directory.

    Args:
        source: The source with the paths.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.utils.iter import DirFilter
    >>> it = DirFilter([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")])
    >>> it
    DirFilter()

    ```
    """

    def __init__(self, source: Iterable[Path]) -> None:
        self._source = source

    def __iter__(self) -> Iterator[Path]:
        for path in self._source:
            if path.is_dir():
                yield path

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"source": str_indent(self._source)}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class FileFilter(Iterable[Path]):
    r"""Implement an iterable to keep only the files.

    Args:
        source: The source with the paths.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.utils.iter import FileFilter
    >>> it = FileFilter([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")])
    >>> it
    FileFilter()

    ```
    """

    def __init__(self, source: Iterable[Path]) -> None:
        self._source = source

    def __iter__(self) -> Iterator[Path]:
        for path in self._source:
            if path.is_file():
                yield path

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"source": str_indent(self._source)}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class PathLister(Iterable[Path]):
    r"""Implement an iterable to list paths.

    Args:
        source: The source with the paths.
        pattern: A glob pattern, to return only the matching paths.
        deterministic: If ``True``, the paths are returned in a
            deterministic order.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.utils.iter import PathLister
    >>> it = PathLister([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")])
    >>> it
    PathLister(pattern=*, deterministic=True)

    ```
    """

    def __init__(
        self,
        source: Iterable[Path],
        pattern: str = "*",
        deterministic: bool = True,
    ) -> None:
        self._source = source
        self._pattern = pattern
        self._deterministic = bool(deterministic)

    def __iter__(self) -> Iterator[Path]:
        for path in self._source:
            paths = path.glob(self._pattern)
            if self._deterministic:
                paths = sorted(paths)
            yield from paths

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(pattern={self._pattern}, "
            f"deterministic={self._deterministic})"
        )

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "pattern": self._pattern,
                    "deterministic": self._deterministic,
                    "source": str_indent(self._source),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
