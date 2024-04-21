r"""Contain code to prepare/preprocess the MultiTHUMOS data.

Information about the MultiTHUMOS dataset can be found in the following
paper:

Every Moment Counts: Dense Detailed Labeling of Actions in Complex
Videos. Yeung S., Russakovsky O., Jin N., Andriluka M., Mori G., Fei-Fei
L. IJCV 2017 (

http://arxiv.org/pdf/1507.05738)

Project page: http://ai.stanford.edu/~syyeung/everymoment.html
"""

from __future__ import annotations

__all__ = ["download_data"]

import logging
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from iden.utils.path import sanitize_path

from arctix.utils.download import download_url_to_file

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

ANNOTATION_URL = "http://ai.stanford.edu/~syyeung/resources/multithumos.zip"


class Column:
    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    END_TIME: str = "end_time"
    SEQUENCE_LENGTH: str = "sequence_length"
    START_TIME: str = "start_time"
    VIDEO: str = "video"
    VIDEO_ID: str = "video_id"


def download_data(path: Path, force_download: bool = False) -> None:
    r"""Download the MultiTHUMOS annotation data.

    Internally, this function downloads the annotations in a temporary
    directory, then extracts the files from the download zip files in
    the temporary directory, and finally moves the extracted files to
    the given path.

    Args:
        path: The path where to store the MultiTHUMOS data.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import download_data
    >>> path = Path("/path/to/data")
    >>> download_data(path)  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    if not is_annotation_path_ready(path) or force_download:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_file = tmp_path.joinpath("multithumos.zip.tmp")
            logger.info(f"downloading MultiTHUMOS annotations data in {zip_file}...")
            download_url_to_file(ANNOTATION_URL, zip_file.as_posix(), progress=True)

            logger.info(f"extracting {zip_file} in {tmp_path}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmp_path)

            logger.info(f"moving extracted files to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            tmp_path.joinpath("multithumos/README").rename(path.joinpath("README"))
            tmp_path.joinpath("multithumos/class_list.txt").rename(path.joinpath("class_list.txt"))
            tmp_path.joinpath("multithumos/annotations").rename(path.joinpath("annotations"))

    logger.info(f"MultiTHUMOS annotation data are available in {path}")


def is_annotation_path_ready(path: Path) -> bool:
    r"""Indicate if the given path contains the MultiTHUMOS annotation
    data.

    Args:
        path: The path to check.

    Returns:
        ``True`` if the path contains the MultiTHUMOS data,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import is_annotation_path_ready
    >>> is_annotation_path_ready(Path("/path/to/data/"))
    False

    ```
    """
    path = sanitize_path(path)
    if not path.joinpath("README").is_file():
        return False
    if not path.joinpath("class_list.txt").is_file():
        return False
    if not path.joinpath("annotations").is_dir():
        return False
    return len(tuple(path.joinpath("annotations").glob("*.txt"))) == 65


def parse_annotation_lines(lines: Sequence[str]) -> dict:
    r"""Parse the action annotation lines and returns a dictionary with
    the prepared data.

    Args:
        lines: The lines to parse.

    Returns:
        A dictionary with the sequence of video names, the start
            time and end time of each action.
    """
    videos = []
    start_time = []
    end_time = []
    for line in (string.strip() for string in lines):
        if not line:
            continue
        video, start, end = line.split(" ")
        videos.append(video)
        start_time.append(float(start))
        end_time.append(float(end))
    return {Column.VIDEO: videos, Column.START_TIME: start_time, Column.END_TIME: end_time}


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("multithumos")
    download_data(path)
