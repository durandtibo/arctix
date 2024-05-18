r"""Contain code to prepare/preprocess the MultiTHUMOS data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/multithumos/`.
"""

from __future__ import annotations

__all__ = [
    "ANNOTATION_FILENAMES",
    "ANNOTATION_URL",
    "Column",
    "MetadataKeys",
    "download_data",
    "fetch_data",
    "filter_by_split",
    "generate_split_column",
    "group_by_sequence",
    "is_annotation_path_ready",
    "load_annotation_file",
    "load_data",
    "parse_annotation_lines",
    "prepare_data",
    "to_array",
    "to_list",
]

import logging
import zipfile
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from iden.utils.path import sanitize_path

from arctix.transformer import dataframe as td
from arctix.utils.dataframe import generate_vocabulary
from arctix.utils.download import download_url_to_file
from arctix.utils.iter import FileFilter, PathLister
from arctix.utils.mapping import convert_to_dict_of_flat_lists
from arctix.utils.masking import convert_sequences_to_array, generate_mask_from_lengths

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

ANNOTATION_URL = "http://ai.stanford.edu/~syyeung/resources/multithumos.zip"

ANNOTATION_FILENAMES = [
    "README",
    "class_list.txt",
    "annotations/Fall.txt",
    "annotations/OneRaisedArmCelebrate.txt",
    "annotations/VolleyballBlock.txt",
    "annotations/OneHandedCatch.txt",
    "annotations/LongJump.txt",
    "annotations/Throw.txt",
    "annotations/HighFive.txt",
    "annotations/DiscusRelease.txt",
    "annotations/TwoRaisedArmCelebrate.txt",
    "annotations/PatPerson.txt",
    "annotations/CricketBowling.txt",
    "annotations/StandUp.txt",
    "annotations/BasketballDunk.txt",
    "annotations/Stand.txt",
    "annotations/VolleyballBump.txt",
    "annotations/HammerThrowRelease.txt",
    "annotations/Diving.txt",
    "annotations/BodyRoll.txt",
    "annotations/ThrowDiscus.txt",
    "annotations/BodyContract.txt",
    "annotations/TalkToCamera.txt",
    "annotations/ShotPutBend.txt",
    "annotations/BaseballPitch.txt",
    "annotations/VolleyballSet.txt",
    "annotations/BodyTurn.txt",
    "annotations/DiscusWindUp.txt",
    "annotations/Jump.txt",
    "annotations/FistPump.txt",
    "annotations/Hug.txt",
    "annotations/PickUp.txt",
    "annotations/SoccerPenalty.txt",
    "annotations/BasketballShot.txt",
    "annotations/HammerThrowWindUp.txt",
    "annotations/CloseUpTalkToCamera.txt",
    "annotations/BasketballBlock.txt",
    "annotations/CliffDiving.txt",
    "annotations/BasketballGuard.txt",
    "annotations/BodyBend.txt",
    "annotations/VolleyballSpiking.txt",
    "annotations/FrisbeeCatch.txt",
    "annotations/PoleVault.txt",
    "annotations/Squat.txt",
    "annotations/HighJump.txt",
    "annotations/Shotput.txt",
    "annotations/BasketballDribble.txt",
    "annotations/ClapHands.txt",
    "annotations/CleanAndJerk.txt",
    "annotations/BasketballPass.txt",
    "annotations/Run.txt",
    "annotations/GolfSwing.txt",
    "annotations/JavelinThrow.txt",
    "annotations/Walk.txt",
    "annotations/Sit.txt",
    "annotations/Drop.txt",
    "annotations/NoHuman.txt",
    "annotations/TwoHandedCatch.txt",
    "annotations/CricketShot.txt",
    "annotations/Billiards.txt",
    "annotations/HammerThrow.txt",
    "annotations/WeightliftingJerk.txt",
    "annotations/VolleyballServe.txt",
    "annotations/HammerThrowSpin.txt",
    "annotations/WeightliftingClean.txt",
    "annotations/TennisSwing.txt",
    "annotations/PoleVaultPlantPole.txt",
]


class Column:
    r"""Indicate the column names."""

    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    END_TIME: str = "end_time"
    SEQUENCE_LENGTH: str = "sequence_length"
    SPLIT: str = "split"
    START_TIME: str = "start_time"
    START_TIME_DIFF: str = "start_time_diff"
    VIDEO: str = "video"
    VIDEO_ID: str = "video_id"


class MetadataKeys:
    r"""Indicate the metadata keys."""

    VOCAB_ACTION: str = "vocab_action"


def fetch_data(path: Path, force_download: bool = False) -> pl.DataFrame:
    r"""Download and load the data for Breakfast dataset.

    Args:
        path: The path where to store the downloaded data.
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
    return load_data(path)


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
            for filename in ANNOTATION_FILENAMES:
                dst = path.joinpath(filename)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.joinpath(f"multithumos/{filename}").rename(dst)

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
    return all(path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def load_data(path: Path) -> pl.DataFrame:
    r"""Load all the annotations in a DataFrame.

    Args:
        path: The directory where the dataset annotations are stored.
        remove_duplicate: If ``True``, the duplicate rows are removed.

    Returns:
        The annotations in a DataFrame.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import load_data
    >>> data = load_data(Path("/path/to/data/multithumos/"))  # doctest: +SKIP

    ```
    """
    paths = FileFilter(PathLister([sanitize_path(path)], pattern="annotations/*.txt"))
    annotations = list(map(load_annotation_file, paths))
    data = convert_to_dict_of_flat_lists(annotations)
    data = pl.DataFrame(data)
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.VIDEO, Column.START_TIME]),
            td.SortColumns(),
        ]
    )
    return transformer.transform(data)


def load_annotation_file(path: Path) -> dict[str, list]:
    r"""Load the annotation data from a text file.

    Args:
        path: The file path to the annotation data.

    Returns:
        A dictionary with the action, the start time, and end time
            of each action.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import load_annotation_file
    >>> data = load_annotation_file(
    ...     Path("/path/to/data/multithumos/annotations/BasketballBlock.txt")
    ... )  # doctest: +SKIP

    ```
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


def prepare_data(frame: pl.DataFrame, split: str = "all") -> tuple[pl.DataFrame, dict]:
    r"""Prepare the data.

    Args:
        frame: The raw DataFrame.
        split: The dataset split. By default, the union of all the
            dataset splits is used.

    Returns:
        A tuple containing the prepared data and the metadata.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, prepare_data
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: ["video_validation_1", "video_test_2", "video_validation_1", "video_test_2"],
    ...         Column.START_TIME: [72.80, 44.00, 1.50, 17.57],
    ...         Column.END_TIME: [76.40, 50.90, 5.40, 18.33],
    ...         Column.ACTION: ["dribble", "dribble", "dribble", "guard"],
    ...     }
    ... )
    >>> data, metadata = prepare_data(frame)
    >>> data
    shape: (4, 7)
    ┌─────────┬───────────┬──────────┬────────────┬────────────┬─────────────────┬────────────────────┐
    │ action  ┆ action_id ┆ end_time ┆ split      ┆ start_time ┆ start_time_diff ┆ video              │
    │ ---     ┆ ---       ┆ ---      ┆ ---        ┆ ---        ┆ ---             ┆ ---                │
    │ str     ┆ i64       ┆ f64      ┆ str        ┆ f64        ┆ f64             ┆ str                │
    ╞═════════╪═══════════╪══════════╪════════════╪════════════╪═════════════════╪════════════════════╡
    │ guard   ┆ 1         ┆ 18.33    ┆ test       ┆ 17.57      ┆ 0.0             ┆ video_test_2       │
    │ dribble ┆ 0         ┆ 50.9     ┆ test       ┆ 44.0       ┆ 26.43           ┆ video_test_2       │
    │ dribble ┆ 0         ┆ 5.4      ┆ validation ┆ 1.5        ┆ 0.0             ┆ video_validation_1 │
    │ dribble ┆ 0         ┆ 76.4     ┆ validation ┆ 72.8       ┆ 71.3            ┆ video_validation_1 │
    └─────────┴───────────┴──────────┴────────────┴────────────┴─────────────────┴────────────────────┘
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
            td.TimeDiff(
                group_cols=[Column.VIDEO],
                time_col=Column.START_TIME,
                time_diff_col=Column.START_TIME_DIFF,
            ),
            td.Sort(columns=[Column.VIDEO, Column.START_TIME]),
            td.Cast(columns=[Column.START_TIME, Column.END_TIME], dtype=pl.Float64),
            td.StripChars(columns=[Column.ACTION, Column.VIDEO]),
            td.TokenToIndex(
                vocab=vocab_action, token_column=Column.ACTION, index_column=Column.ACTION_ID
            ),
            td.Function(generate_split_column),
            td.Function(partial(filter_by_split, split=split)),
            td.SortColumns(),
        ]
    )
    out = transformer.transform(frame)
    return out, {MetadataKeys.VOCAB_ACTION: vocab_action}


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


def filter_by_split(frame: pl.DataFrame, split: str = "all") -> pl.DataFrame:
    r"""Filter the DataFrame to keep only the rows associated to a
    dataset split.

    Args:
        frame: The DataFrame to filter.
        split: The dataset split. By default, the union of all the
            dataset splits is used.

    Returns:
        The filtered DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, filter_by_split
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: [
    ...             "video_test_1",
    ...             "video_test_1",
    ...             "video_test_1",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...         ],
    ...         Column.START_TIME: [1.50, 17.57, 79.30, 2.97, 4.54, 20.22, 27.42],
    ...         Column.END_TIME: [5.40, 18.33, 83.90, 3.60, 5.07, 20.49, 30.23],
    ...         Column.ACTION: [
    ...             "dribble",
    ...             "guard",
    ...             "dribble",
    ...             "guard",
    ...             "guard",
    ...             "guard",
    ...             "shoot",
    ...         ],
    ...         Column.ACTION_ID: [1, 0, 1, 0, 0, 0, 2],
    ...         Column.SPLIT: [
    ...             "test",
    ...             "test",
    ...             "test",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...         ],
    ...     },
    ... )
    >>> data = filter_by_split(frame, split='test')
    >>> data
    shape: (3, 6)
    ┌──────────────┬────────────┬──────────┬─────────┬───────────┬───────┐
    │ video        ┆ start_time ┆ end_time ┆ action  ┆ action_id ┆ split │
    │ ---          ┆ ---        ┆ ---      ┆ ---     ┆ ---       ┆ ---   │
    │ str          ┆ f64        ┆ f64      ┆ str     ┆ i64       ┆ str   │
    ╞══════════════╪════════════╪══════════╪═════════╪═══════════╪═══════╡
    │ video_test_1 ┆ 1.5        ┆ 5.4      ┆ dribble ┆ 1         ┆ test  │
    │ video_test_1 ┆ 17.57      ┆ 18.33    ┆ guard   ┆ 0         ┆ test  │
    │ video_test_1 ┆ 79.3       ┆ 83.9     ┆ dribble ┆ 1         ┆ test  │
    └──────────────┴────────────┴──────────┴─────────┴───────────┴───────┘

    ```
    """
    splits = {split}
    if split == "all":
        splits = {"validation", "test"}
    return frame.filter(pl.col(Column.SPLIT).is_in(splits))


def group_by_sequence(frame: pl.DataFrame) -> pl.DataFrame:
    r"""Group the DataFrame by sequences of actions.

    Args:
        frame: The input DataFrame.

    Returns:
        The DataFrame after the grouping.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, group_by_sequence
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: [
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...         ],
    ...         Column.START_TIME: [1.50, 17.57, 79.30, 2.97, 4.54, 20.22, 27.42],
    ...         Column.START_TIME_DIFF: [0.0, 16.07, 61.73, 0.0, 1.57, 15.68, 7.20],
    ...         Column.END_TIME: [5.40, 18.33, 83.90, 3.60, 5.07, 20.49, 30.23],
    ...         Column.ACTION: [
    ...             "dribble",
    ...             "guard",
    ...             "dribble",
    ...             "guard",
    ...             "guard",
    ...             "guard",
    ...             "shoot",
    ...         ],
    ...         Column.ACTION_ID: [1, 0, 1, 0, 0, 0, 2],
    ...         Column.SPLIT: [
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...         ],
    ...     },
    ... )
    >>> groups = group_by_sequence(frame)
    >>> groups
    shape: (2, 8)
    ┌────────────┬────────────┬────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
    │ action     ┆ action_id  ┆ end_time   ┆ sequence_ ┆ split     ┆ start_tim ┆ start_tim ┆ video     │
    │ ---        ┆ ---        ┆ ---        ┆ length    ┆ ---       ┆ e         ┆ e_diff    ┆ ---       │
    │ list[str]  ┆ list[i64]  ┆ list[f64]  ┆ ---       ┆ str       ┆ ---       ┆ ---       ┆ str       │
    │            ┆            ┆            ┆ i64       ┆           ┆ list[f64] ┆ list[f64] ┆           │
    ╞════════════╪════════════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
    │ ["dribble" ┆ [1, 0, 1]  ┆ [5.4,      ┆ 3         ┆ validatio ┆ [1.5,     ┆ [0.0,     ┆ video_val │
    │ , "guard", ┆            ┆ 18.33,     ┆           ┆ n         ┆ 17.57,    ┆ 16.07,    ┆ idation_1 │
    │ "dribble"… ┆            ┆ 83.9]      ┆           ┆           ┆ 79.3]     ┆ 61.73]    ┆           │
    │ ["guard",  ┆ [0, 0, …   ┆ [3.6,      ┆ 4         ┆ validatio ┆ [2.97,    ┆ [0.0,     ┆ video_val │
    │ "guard", … ┆ 2]         ┆ 5.07, …    ┆           ┆ n         ┆ 4.54, …   ┆ 1.57, …   ┆ idation_2 │
    │ "shoot"]   ┆            ┆ 30.23]     ┆           ┆           ┆ 27.42]    ┆ 7.2]      ┆           │
    └────────────┴────────────┴────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

    ```
    """
    data = frame.group_by([Column.VIDEO]).agg(
        pl.first(Column.SPLIT),
        pl.col(Column.ACTION),
        pl.col(Column.ACTION_ID),
        pl.col(Column.START_TIME),
        pl.col(Column.START_TIME_DIFF),
        pl.col(Column.END_TIME),
        pl.len().cast(pl.Int64).alias(Column.SEQUENCE_LENGTH),
    )
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.VIDEO]),
            td.SortColumns(),
        ]
    )
    return transformer.transform(data)


def to_array(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    r"""Convert a DataFrame to a dictionary of arrays.

    Args:
        frame: The input DataFrame.

    Returns:
        The dictionary of arrays.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, to_array
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: [
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...         ],
    ...         Column.START_TIME: [1.0, 17.0, 79.0, 2.0, 4.0, 20.0, 27.0],
    ...         Column.START_TIME_DIFF: [0.0, 16.07, 61.73, 0.0, 1.57, 15.68, 7.20],
    ...         Column.END_TIME: [5.0, 18.0, 83.0, 3.0, 5.0, 20.0, 30.0],
    ...         Column.ACTION: [
    ...             "dribble",
    ...             "guard",
    ...             "dribble",
    ...             "guard",
    ...             "guard",
    ...             "guard",
    ...             "shoot",
    ...         ],
    ...         Column.ACTION_ID: [1, 0, 1, 0, 0, 0, 2],
    ...         Column.SPLIT: [
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...         ],
    ...     },
    ... )
    >>> arrays = to_array(frame)
    >>> arrays
    {'action': masked_array(
      data=[['dribble', 'guard', 'dribble', --],
            ['guard', 'guard', 'guard', 'shoot']],
      mask=[[False, False, False,  True],
            [False, False, False, False]],
      fill_value='N/A',
      dtype='<U7'),
      'action_id': masked_array(
      data=[[1, 0, 1, --],
            [0, 0, 0, 2]],
      mask=[[False, False, False,  True],
            [False, False, False, False]],
      fill_value=999999),
      'end_time': masked_array(
      data=[[5.0, 18.0, 83.0, --],
            [3.0, 5.0, 20.0, 30.0]],
      mask=[[False, False, False,  True],
            [False, False, False, False]],
      fill_value=1e+20),
      'sequence_length': array([3, 4]),
      'split': array(['validation', 'validation'], dtype='<U10'),
      'start_time': masked_array(
      data=[[1.0, 17.0, 79.0, --],
            [2.0, 4.0, 20.0, 27.0]],
      mask=[[False, False, False,  True],
            [False, False, False, False]],
      fill_value=1e+20),
      'start_time_diff': masked_array(
      data=[[0.0, 16.07, 61.73, --],
            [0.0, 1.57, 15.68, 7.2]],
      mask=[[False, False, False,  True],
            [False, False, False, False]],
      fill_value=1e+20)}

    ```
    """
    groups = group_by_sequence(frame)
    lengths = groups.get_column(Column.SEQUENCE_LENGTH).to_numpy()
    mask = generate_mask_from_lengths(lengths)
    return {
        Column.ACTION: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION).to_list(),
                max_len=mask.shape[1],
                dtype=np.object_,
                padded_value="N/A",
            ).astype(str),
            mask=mask,
        ),
        Column.ACTION_ID: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_ID).to_list(),
                dtype=int,
                max_len=mask.shape[1],
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.END_TIME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.END_TIME).to_list(),
                dtype=np.float64,
                max_len=mask.shape[1],
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.SEQUENCE_LENGTH: lengths.astype(np.int64),
        Column.SPLIT: groups.get_column(Column.SPLIT).to_numpy().astype(str),
        Column.START_TIME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME).to_list(),
                dtype=np.float64,
                max_len=mask.shape[1],
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.START_TIME_DIFF: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME_DIFF).to_list(),
                dtype=np.float64,
                max_len=mask.shape[1],
                padded_value=-1.0,
            ),
            mask=mask,
        ),
    }


def to_list(frame: pl.DataFrame) -> dict[str, list]:
    r"""Convert a DataFrame to a dictionary of lists.

    Args:
        frame: The input DataFrame.

    Returns:
        The dictionary of lists.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.multithumos import Column, to_list
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.VIDEO: [
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_1",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...             "video_validation_2",
    ...         ],
    ...         Column.START_TIME: [1.0, 17.0, 79.0, 2.0, 4.0, 20.0, 27.0],
    ...         Column.START_TIME_DIFF: [0.0, 16.07, 61.73, 0.0, 1.57, 15.68, 7.20],
    ...         Column.END_TIME: [5.0, 18.0, 83.0, 3.0, 5.0, 20.0, 30.0],
    ...         Column.ACTION: [
    ...             "dribble",
    ...             "guard",
    ...             "dribble",
    ...             "guard",
    ...             "guard",
    ...             "guard",
    ...             "shoot",
    ...         ],
    ...         Column.ACTION_ID: [1, 0, 1, 0, 0, 0, 2],
    ...         Column.SPLIT: [
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...             "validation",
    ...         ],
    ...     },
    ... )
    >>> data_list = to_list(frame)
    >>> data_list
    {'action': [['dribble', 'guard', 'dribble'], ['guard', 'guard', 'guard', 'shoot']],
     'action_id': [[1, 0, 1], [0, 0, 0, 2]],
     'end_time': [[5.0, 18.0, 83.0], [3.0, 5.0, 20.0, 30.0]],
     'sequence_length': [3, 4],
     'split': ['validation', 'validation'],
     'start_time': [[1.0, 17.0, 79.0], [2.0, 4.0, 20.0, 27.0]],
     'start_time_diff': [[0.0, 16.07, 61.73], [0.0, 1.57, 15.68, 7.2]],
     'video': ['video_validation_1', 'video_validation_2']}

    ```
    """
    return group_by_sequence(frame).to_dict(as_series=False)


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("multithumos")
    raw_data = fetch_data(path)
    logger.info(f"data_raw:\n{raw_data}")
    data, metadata = prepare_data(raw_data, split="validation")
    logger.info(f"data:\n{data}")
    logger.info(f"metadata:\n{metadata}")
