r"""Contain utility functions to download data assets."""

from __future__ import annotations

__all__ = ["download_drive_file", "download_url_to_file"]

from pathlib import Path
from typing import Any

from iden.io.utils import generate_unique_tmp_path
from iden.utils.path import sanitize_path

from arctix.utils.imports import (
    check_gdown,
    check_requests,
    is_gdown_available,
    is_requests_available,
    is_tqdm_available,
)

if is_gdown_available():
    import gdown
else:  # pragma: no cover
    gdown = None

if is_requests_available():
    import requests
else:  # pragma: no cover
    requests = None

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from arctix.utils.noop import tqdm


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


def download_url_to_file(
    url: str, dst: Path | str, progress: bool = True, timeout: float = 10.0
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url: The URL of the object to download
        dst: The path where to store the downloaded file.
        progress: If ``True``, it displays a progress bar.
        timeout: The number of second to wait until to time out.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.utils.download import download_url_to_file
    >>> download_url_to_file(
    ...     'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    ...     Path('/tmp/temporary_file'),
    ... )  # doctest: +SKIP

    ```
    """
    check_requests()
    dst = sanitize_path(dst)
    dst.parent.mkdir(exist_ok=True, parents=True)

    # Save to tmp, then commit by moving the file in case the job gets
    # interrupted while writing the file
    tmp_dst = generate_unique_tmp_path(dst)

    with (
        requests.get(url, stream=True, timeout=timeout) as response,
        Path.open(tmp_dst, mode="wb") as file,
    ):
        chunks = response.iter_content(chunk_size=1024)
        if progress:
            chunks = tqdm(chunks, desc="downloading URL to file")
        for chunk in chunks:
            file.write(chunk)

    tmp_dst.rename(dst)
