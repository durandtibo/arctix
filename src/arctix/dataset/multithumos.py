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

import polars as pl
from iden.utils.path import sanitize_path

from arctix.transformer import dataframe as td
from arctix.utils.dataframe import drop_duplicates, generate_vocabulary
from arctix.utils.download import download_url_to_file
from arctix.utils.iter import FileFilter, PathLister
from arctix.utils.mapping import convert_to_dict_of_flat_lists

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

ANNOTATION_URL = "http://ai.stanford.edu/~syyeung/resources/multithumos.zip"


class Column:
    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    END_TIME: str = "end_time"
    SEQUENCE_LENGTH: str = "sequence_length"
    SPLIT: str = "split"
    START_TIME: str = "start_time"
    VIDEO: str = "video"
    VIDEO_ID: str = "video_id"


def fetch_data(
    path: Path, remove_duplicate: bool = True, force_download: bool = False
) -> pl.DataFrame:
    r"""Download and load the data for Breakfast dataset.

    Args:
        path: The path where to store the downloaded data.
        remove_duplicate: If ``True``, the duplicate examples are
            removed.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Returns:
        The data in a DataFrame

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import fetch_data
    >>> data = fetch_data(Path("/path/to/data/multithumos/"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    download_data(path, force_download)
    return load_data(path, remove_duplicate)


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


def load_data(path: Path, remove_duplicate: bool = True) -> pl.DataFrame:
    r"""Load all the annotations in a DataFrame.

    Args:
        path: The directory where the dataset annotations are stored.
        remove_duplicate: If ``True``, the duplicate rows are removed.

    Returns:
        The annotations in a DataFrame.
    """
    paths = FileFilter(PathLister([sanitize_path(path)], pattern="annotations/*.txt"))
    annotations = list(map(load_annotation_file, paths))
    data = convert_to_dict_of_flat_lists(annotations)
    data = pl.DataFrame(data)
    if remove_duplicate:
        data = drop_duplicates(data)
    if data.select(pl.len()).item():
        data = data.sort(by=[Column.VIDEO, Column.START_TIME])
    return data


def load_annotation_file(path: Path) -> dict[str, list]:
    r"""Load the annotation data from a text file.

    Args:
        path: The file path to the annotation data.

    Returns:
        A dictionary with the action, the start time, and end time
            of each action.
    """
    path = sanitize_path(path)
    if path.suffix != ".txt":
        msg = (
            "Incorrect file extension. This function can only parse `.txt` files "
            f"but received {path.suffix}"
        )
        raise ValueError(msg)
    logger.info(f"Reading {path}...")
    with Path.open(path) as file:
        lines = [x.strip() for x in file.readlines()]

    annotation = parse_annotation_lines(lines)
    annotation[Column.ACTION] = [path.stem] * len(annotation[Column.VIDEO])
    return annotation


def parse_annotation_lines(lines: Sequence[str]) -> dict:
    r"""Parse the action annotation lines and returns a dictionary with
    the prepared data.

    Args:
        lines: The lines to parse.

    Returns:
        A dictionary with the sequence of video names, the start
            time and end time of each action.

    Example usage:

    ```pycon

    >>> from arctix.dataset.multithumos import parse_annotation_lines
    >>> out = parse_annotation_lines(
    ...     [
    ...         "video_validation_0000266 72.80 76.40",
    ...         "video_validation_0000681 44.00 50.90",
    ...         "video_validation_0000682 1.50 5.40",
    ...         "video_validation_0000682 79.30 83.90",
    ...     ]
    ... )
    >>> out
    {'video': ['video_validation_0000266', 'video_validation_0000681', 'video_validation_0000682', 'video_validation_0000682'],
     'start_time': [72.8, 44.0, 1.5, 79.3],
     'end_time': [76.4, 50.9, 5.4, 83.9]}

    ```
    """
    videos = []
    start_time = []
    end_time = []
    for line in (item.strip() for item in lines):
        if not line:
            continue
        video, start, end = line.split(" ")
        videos.append(video)
        start_time.append(float(start))
        end_time.append(float(end))
    return {Column.VIDEO: videos, Column.START_TIME: start_time, Column.END_TIME: end_time}


def prepare_data(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    r"""Prepare the data.

    Args:
        frame: The raw DataFrame.

    Returns:
        A tuple containing the prepared data and the metadata.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, prepare_data
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: ["video_validation_1", "video_test_2", "video_validation_3", "video_test_4"],
    ...         Column.START_TIME: [72.80, 44.00, 1.50, 17.57],
    ...         Column.END_TIME: [76.40, 50.90, 5.40, 18.33],
    ...         Column.ACTION: ["dribble", "dribble", "dribble", "guard"],
    ...     }
    ... )
    >>> data, metadata = prepare_data(frame)
    >>> data
    shape: (4, 6)
    ┌────────────────────┬────────────┬───────────┬─────────┬───────────┬────────────┐
    │ video              ┆ start_time ┆ end_time  ┆ action  ┆ action_id ┆ split      │
    │ ---                ┆ ---        ┆ ---       ┆ ---     ┆ ---       ┆ ---        │
    │ str                ┆ f32        ┆ f32       ┆ str     ┆ i64       ┆ str        │
    ╞════════════════════╪════════════╪═══════════╪═════════╪═══════════╪════════════╡
    │ video_test_2       ┆ 44.0       ┆ 50.900002 ┆ dribble ┆ 0         ┆ test       │
    │ video_test_4       ┆ 17.57      ┆ 18.33     ┆ guard   ┆ 1         ┆ test       │
    │ video_validation_1 ┆ 72.800003  ┆ 76.400002 ┆ dribble ┆ 0         ┆ validation │
    │ video_validation_3 ┆ 1.5        ┆ 5.4       ┆ dribble ┆ 0         ┆ validation │
    └────────────────────┴────────────┴───────────┴─────────┴───────────┴────────────┘
    >>> metadata
    {'vocab_action': Vocabulary(
      counter=Counter({'dribble': 3, 'guard': 1}),
      index_to_token=('dribble', 'guard'),
      token_to_index={'dribble': 0, 'guard': 1},
    )}

    ```
    """
    vocab_action = generate_vocabulary(frame, col=Column.ACTION).sort_by_count()
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.VIDEO, Column.START_TIME]),
            td.Cast(columns=[Column.START_TIME, Column.END_TIME], dtype=pl.Float32),
            td.StripChars(columns=[Column.ACTION, Column.VIDEO]),
            td.TokenToIndex(
                vocab=vocab_action, token_column=Column.ACTION, index_column=Column.ACTION_ID
            ),
            td.Function(generate_split_column),
        ]
    )
    out = transformer.transform(frame)
    return out, {"vocab_action": vocab_action}


def generate_split_column(frame: pl.DataFrame) -> pl.DataFrame:
    r"""Generate the split column from the video name column.

    Args:
        frame: The input DataFrame with the video name column.

    Returns:
        The output DataFrame with the additional split column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, generate_split_column
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: ["video_validation_1", "video_test_2", "video_validation_3", "video_test_4"],
    ...         Column.ACTION_ID: [0, 2, 5, 1],
    ...     }
    ... )
    >>> out = generate_split_column(frame)
    >>> out
    shape: (4, 3)
    ┌────────────────────┬───────────┬────────────┐
    │ video              ┆ action_id ┆ split      │
    │ ---                ┆ ---       ┆ ---        │
    │ str                ┆ i64       ┆ str        │
    ╞════════════════════╪═══════════╪════════════╡
    │ video_validation_1 ┆ 0         ┆ validation │
    │ video_test_2       ┆ 2         ┆ test       │
    │ video_validation_3 ┆ 5         ┆ validation │
    │ video_test_4       ┆ 1         ┆ test       │
    └────────────────────┴───────────┴────────────┘

    ```
    """
    return frame.with_columns(
        pl.col(Column.VIDEO).str.split_exact(by="_", n=3).struct[1].alias(Column.SPLIT)
    )


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("multithumos")
    raw_data = fetch_data(path)
    data, metadata = prepare_data(raw_data)
    logger.info(f"data:\n{data}")
    logger.info(f"metadata:\n{metadata}")
