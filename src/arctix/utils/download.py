r"""Contain utility functions to download data assets."""

from __future__ import annotations

__all__ = ["download_drive_file"]

from typing import TYPE_CHECKING, Any

from arctix.utils.imports import check_gdown, is_gdown_available

if TYPE_CHECKING:
    from pathlib import Path

if is_gdown_available():
    import gdown
else:  # pragma: no cover
    gdown = None


def download_drive_file(url: str, path: Path, *args: Any, **kwargs: Any) -> None:
    r"""Download a file from Google Drive.

    Args:
        url: The Google Drive URL.
        path: The path where to store the downloaded file.
        *args: See the documentation of ``gdown.download``.
        **kwargs: See the documentation of ``gdown.download``.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.utils.download import download_drive_file
    >>> download_drive_file(
    ...     "https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E",
    ...     Path("/path/to/data.tar.gz"),
    ...     quiet=False,
    ...     fuzzy=True,
    ... )  # doctest: +SKIP

    ```
    """
    check_gdown()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        # Save to tmp, then commit by moving the file in case the job gets
        # interrupted while writing the file
        tmp_path = path.with_name(f"{path.name}.tmp")
        gdown.download(url, tmp_path.as_posix(), *args, **kwargs)
        tmp_path.rename(path)
