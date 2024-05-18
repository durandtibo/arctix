r"""Contain code to download and prepare the Breakfast data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/breakfast/`.
"""

from __future__ import annotations

__all__ = [
    "COOKING_ACTIVITIES",
    "Column",
    "DATASET_SPLITS",
    "MetadataKeys",
    "NUM_COOKING_ACTIVITIES",
    "URLS",
    "download_data",
    "fetch_data",
    "filter_by_split",
    "group_by_sequence",
    "load_annotation_file",
    "load_data",
    "parse_annotation_lines",
    "prepare_data",
    "to_array",
    "to_list",
]

import logging
import tarfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from iden.utils.path import sanitize_path

from arctix.transformer import dataframe as td
from arctix.utils.dataframe import drop_duplicates, generate_vocabulary
from arctix.utils.download import download_drive_file
from arctix.utils.iter import FileFilter, PathLister
from arctix.utils.mapping import convert_to_dict_of_flat_lists
from arctix.utils.masking import convert_sequences_to_array, generate_mask_from_lengths

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

URLS = {
    "segmentation_coarse": "https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E",
    "segmentation_fine": "https://drive.google.com/open?id=1Alg_xjefEFOOpO_6_RnelWiNqbJlKhVF",
}
COOKING_ACTIVITIES = (
    "cereals",
    "coffee",
    "friedegg",
    "juice",
    "milk",
    "pancake",
    "salat",
    "sandwich",
    "scrambledegg",
    "tea",
)
NUM_COOKING_ACTIVITIES = {
    "cereals": 214,
    "coffee": 100,
    "friedegg": 198,
    "juice": 187,
    "milk": 224,
    "pancake": 173,
    "salat": 185,
    "sandwich": 197,
    "scrambledegg": 188,
    "tea": 223,
}

PART1 = tuple(f"P{i:02d}" for i in range(3, 16))
PART2 = tuple(f"P{i:02d}" for i in range(16, 29))
PART3 = tuple(f"P{i:02d}" for i in range(29, 42))
PART4 = tuple(f"P{i:02d}" for i in range(42, 55))

DATASET_SPLITS = {
    "all": sorted(PART1 + PART2 + PART3 + PART4),
    "minitrain1": sorted(PART2 + PART3),
    "minitrain2": sorted(PART3 + PART4),
    "minitrain3": sorted(PART1 + PART4),
    "minitrain4": sorted(PART1 + PART2),
    "minival1": sorted(PART4),
    "minival2": sorted(PART1),
    "minival3": sorted(PART2),
    "minival4": sorted(PART3),
    "test1": sorted(PART1),
    "test2": sorted(PART2),
    "test3": sorted(PART3),
    "test4": sorted(PART4),
    "train1": sorted(PART2 + PART3 + PART4),
    "train2": sorted(PART1 + PART3 + PART4),
    "train3": sorted(PART1 + PART2 + PART4),
    "train4": sorted(PART1 + PART2 + PART3),
}


class Column:
    r"""Indicate the column names."""

    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    COOKING_ACTIVITY: str = "cooking_activity"
    COOKING_ACTIVITY_ID: str = "cooking_activity_id"
    END_TIME: str = "end_time"
    PERSON: str = "person"
    PERSON_ID: str = "person_id"
    START_TIME: str = "start_time"
    START_TIME_DIFF: str = "start_time_diff"
    SEQUENCE_LENGTH: str = "sequence_length"


class MetadataKeys:
    r"""Indicate the metadata keys."""

    VOCAB_ACTION: str = "vocab_action"
    VOCAB_ACTIVITY: str = "vocab_activity"
    VOCAB_PERSON: str = "vocab_person"


def fetch_data(
    path: Path, name: str, remove_duplicate: bool = True, force_download: bool = False
) -> pl.DataFrame:
    r"""Download and load the data for Breakfast dataset.

    Args:
        path: The path where to store the downloaded data.
        name: The name of the dataset. The valid names are
            ``'segmentation_coarse'`` and ``'segmentation_fine'``.
        remove_duplicate: If ``True``, the duplicate examples are
            removed.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Returns:
        The data in a DataFrame

    Raises:
        RuntimeError: if the name is incorrect

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.breakfast import fetch_data
    >>> data = fetch_data(
    ...     Path("/path/to/data/breakfast/"), "segmentation_coarse"
    ... )  # doctest: +SKIP

    ```
    """
    if name not in (valid_names := set(URLS.keys())):
        msg = f"Incorrect name: {name}. Valid names are: {valid_names}"
        raise RuntimeError(msg)
    path = sanitize_path(path)
    download_data(path, force_download)
    return load_data(path.joinpath(name), remove_duplicate)


def download_data(path: Path, force_download: bool = False) -> None:
    r"""Download the Breakfast annotations.

    Args:
        path: The path where to store the downloaded data.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.breakfast import download_data
    >>> download_data(Path("/path/to/data/breakfast/"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    logger.info(f"Downloading Breakfast dataset annotations in {path}...")
    for name, url in URLS.items():
        if not path.joinpath(name).is_dir() or force_download:
            tar_file = path.joinpath(f"{name}.tar.gz")
            download_drive_file(url, tar_file, quiet=False, fuzzy=True)
            tarfile.open(tar_file).extractall(path)  # noqa: S202
            tar_file.unlink(missing_ok=True)


def load_data(path: Path, remove_duplicate: bool = True) -> pl.DataFrame:
    r"""Load all the annotations in a DataFrame.

    Args:
        path: The directory where the dataset annotations are stored.
        remove_duplicate: If ``True``, the duplicate rows are removed.

    Returns:
        The annotations in a DataFrame.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.breakfast import load_data
    >>> data = load_data(Path("/path/to/data/breakfast/"))  # doctest: +SKIP

    ```
    """
    paths = FileFilter(PathLister([sanitize_path(path)], pattern="**/*.txt"))
    annotations = list(map(load_annotation_file, paths))
    data = convert_to_dict_of_flat_lists(annotations)
    data = pl.DataFrame(data)
    if remove_duplicate:
        data = drop_duplicates(data)
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.COOKING_ACTIVITY, Column.PERSON, Column.START_TIME]),
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
    >>> from arctix.dataset.breakfast import load_annotation_file
    >>> data = load_annotation_file(Path("/path/to/data/breakfast/segmentation_coarse/cereals/P03_cam01_P03_cereals.txt"))  # doctest: +SKIP

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
    person_id = path.stem.split("_", maxsplit=1)[0]
    cooking_activity = path.stem.rsplit("_", maxsplit=1)[-1]
    annotation[Column.PERSON] = [person_id] * len(lines)
    annotation[Column.COOKING_ACTIVITY] = [cooking_activity] * len(lines)
    return annotation


def parse_annotation_lines(lines: Sequence[str]) -> dict:
    r"""Parse the action annotation lines and returns a dictionary with
    the prepared data.

    Args:
        lines: The lines to parse.

    Returns:
        A dictionary with the sequence of actions, the start
            time and end time of each action.
    """
    actions = []
    start_time = []
    end_time = []
    for line in lines:
        pair_time, action = line.strip().split()
        actions.append(action)
        start, end = pair_time.split("-")
        start_time.append(float(start))
        end_time.append(float(end))
    return {Column.ACTION: actions, Column.START_TIME: start_time, Column.END_TIME: end_time}


def filter_by_split(frame: pl.DataFrame, split: str = "all") -> pl.DataFrame:
    r"""Filter the DataFrame to keep only the rows associated to a
    dataset split.

    Args:
        frame: The DataFrame to filter.
        split: The dataset split. By default, the union of all the
            dataset splits is used.

    Returns:
        The filtered DataFrame.
    """
    persons = DATASET_SPLITS[split]
    return frame.filter(pl.col(Column.PERSON).is_in(persons))


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
    >>> from arctix.dataset.breakfast import Column, group_by_sequence
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION: [
    ...             "SIL",
    ...             "take_bowl",
    ...             "pour_cereals",
    ...             "pour_milk",
    ...             "stir_cereals",
    ...             "SIL",
    ...             "SIL",
    ...             "pour_milk",
    ...             "spoon_powder",
    ...             "SIL",
    ...         ],
    ...         Column.COOKING_ACTIVITY: [
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...         ],
    ...         Column.END_TIME: [
    ...             30.0,
    ...             150.0,
    ...             428.0,
    ...             575.0,
    ...             705.0,
    ...             836.0,
    ...             47.0,
    ...             215.0,
    ...             565.0,
    ...             747.0,
    ...         ],
    ...         Column.PERSON: [
    ...             "P03",
    ...             "P03",
    ...             "P03",
    ...             "P03",
    ...             "P03",
    ...             "P03",
    ...             "P54",
    ...             "P54",
    ...             "P54",
    ...             "P54",
    ...         ],
    ...         Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
    ...     },
    ... )
    >>> data, metadata = prepare_data(frame)
    >>> with pl.Config(tbl_cols=-1):
    ...     data
    shape: (10, 9)
    ┌───────────┬───────────┬──────────┬──────────┬──────────┬────────┬──────────┬──────────┬──────────┐
    │ action    ┆ action_id ┆ cooking_ ┆ cooking_ ┆ end_time ┆ person ┆ person_i ┆ start_ti ┆ start_ti │
    │ ---       ┆ ---       ┆ activity ┆ activity ┆ ---      ┆ ---    ┆ d        ┆ me       ┆ me_diff  │
    │ str       ┆ i64       ┆ ---      ┆ _id      ┆ f64      ┆ str    ┆ ---      ┆ ---      ┆ ---      │
    │           ┆           ┆ str      ┆ ---      ┆          ┆        ┆ i64      ┆ f64      ┆ f64      │
    │           ┆           ┆          ┆ i64      ┆          ┆        ┆          ┆          ┆          │
    ╞═══════════╪═══════════╪══════════╪══════════╪══════════╪════════╪══════════╪══════════╪══════════╡
    │ SIL       ┆ 0         ┆ cereals  ┆ 0        ┆ 30.0     ┆ P03    ┆ 0        ┆ 1.0      ┆ 0.0      │
    │ take_bowl ┆ 2         ┆ cereals  ┆ 0        ┆ 150.0    ┆ P03    ┆ 0        ┆ 31.0     ┆ 30.0     │
    │ pour_cere ┆ 5         ┆ cereals  ┆ 0        ┆ 428.0    ┆ P03    ┆ 0        ┆ 151.0    ┆ 120.0    │
    │ als       ┆           ┆          ┆          ┆          ┆        ┆          ┆          ┆          │
    │ pour_milk ┆ 1         ┆ cereals  ┆ 0        ┆ 575.0    ┆ P03    ┆ 0        ┆ 429.0    ┆ 278.0    │
    │ stir_cere ┆ 3         ┆ cereals  ┆ 0        ┆ 705.0    ┆ P03    ┆ 0        ┆ 576.0    ┆ 147.0    │
    │ als       ┆           ┆          ┆          ┆          ┆        ┆          ┆          ┆          │
    │ SIL       ┆ 0         ┆ cereals  ┆ 0        ┆ 836.0    ┆ P03    ┆ 0        ┆ 706.0    ┆ 130.0    │
    │ SIL       ┆ 0         ┆ milk     ┆ 1        ┆ 47.0     ┆ P54    ┆ 1        ┆ 1.0      ┆ 0.0      │
    │ pour_milk ┆ 1         ┆ milk     ┆ 1        ┆ 215.0    ┆ P54    ┆ 1        ┆ 48.0     ┆ 47.0     │
    │ spoon_pow ┆ 4         ┆ milk     ┆ 1        ┆ 565.0    ┆ P54    ┆ 1        ┆ 216.0    ┆ 168.0    │
    │ der       ┆           ┆          ┆          ┆          ┆        ┆          ┆          ┆          │
    │ SIL       ┆ 0         ┆ milk     ┆ 1        ┆ 747.0    ┆ P54    ┆ 1        ┆ 566.0    ┆ 350.0    │
    └───────────┴───────────┴──────────┴──────────┴──────────┴────────┴──────────┴──────────┴──────────┘
    >>> metadata
    {'vocab_action': Vocabulary(
      counter=Counter({'SIL': 4, 'pour_milk': 2, 'take_bowl': 1, 'stir_cereals': 1, 'spoon_powder': 1, 'pour_cereals': 1}),
      index_to_token=('SIL', 'pour_milk', 'take_bowl', 'stir_cereals', 'spoon_powder', 'pour_cereals'),
      token_to_index={'SIL': 0, 'pour_milk': 1, 'take_bowl': 2, 'stir_cereals': 3, 'spoon_powder': 4, 'pour_cereals': 5},
    ), 'vocab_activity': Vocabulary(
      counter=Counter({'cereals': 6, 'milk': 4}),
      index_to_token=('cereals', 'milk'),
      token_to_index={'cereals': 0, 'milk': 1},
    ), 'vocab_person': Vocabulary(
      counter=Counter({'P03': 6, 'P54': 4}),
      index_to_token=('P03', 'P54'),
      token_to_index={'P03': 0, 'P54': 1},
    )}

    ```
    """
    vocab_action = generate_vocabulary(frame, col=Column.ACTION).sort_by_count()
    vocab_person = generate_vocabulary(frame, col=Column.PERSON).sort_by_count()
    vocab_activity = (
        generate_vocabulary(frame, col=Column.COOKING_ACTIVITY).sort_by_token().sort_by_count()
    )
    transformer = td.Sequential(
        [
            td.TimeDiff(
                group_cols=[Column.COOKING_ACTIVITY, Column.PERSON],
                time_col=Column.START_TIME,
                time_diff_col=Column.START_TIME_DIFF,
            ),
            td.Sort(columns=[Column.COOKING_ACTIVITY, Column.PERSON, Column.START_TIME]),
            td.Cast(columns=[Column.START_TIME, Column.END_TIME], dtype=pl.Float64),
            td.StripChars(columns=[Column.ACTION, Column.PERSON, Column.COOKING_ACTIVITY]),
            td.Function(partial(filter_by_split, split=split)),
            td.TokenToIndex(
                vocab=vocab_action, token_column=Column.ACTION, index_column=Column.ACTION_ID
            ),
            td.TokenToIndex(
                vocab=vocab_person, token_column=Column.PERSON, index_column=Column.PERSON_ID
            ),
            td.TokenToIndex(
                vocab=vocab_activity,
                token_column=Column.COOKING_ACTIVITY,
                index_column=Column.COOKING_ACTIVITY_ID,
            ),
            td.SortColumns(),
        ]
    )
    out = transformer.transform(frame)
    return out, {
        MetadataKeys.VOCAB_ACTION: vocab_action,
        MetadataKeys.VOCAB_ACTIVITY: vocab_activity,
        MetadataKeys.VOCAB_PERSON: vocab_person,
    }


def group_by_sequence(frame: pl.DataFrame) -> pl.DataFrame:
    r"""Group the DataFrame by sequences of actions.

    Args:
        frame: The input DataFrame.

    Returns:
        The DataFrame after the grouping.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.breakfast import Column, group_by_sequence
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION: [
    ...             "SIL",
    ...             "take_bowl",
    ...             "pour_cereals",
    ...             "pour_milk",
    ...             "stir_cereals",
    ...             "SIL",
    ...             "SIL",
    ...             "pour_milk",
    ...             "spoon_powder",
    ...             "SIL",
    ...         ],
    ...         Column.ACTION_ID: [0, 2, 5, 1, 3, 0, 0, 1, 4, 0],
    ...         Column.COOKING_ACTIVITY: [
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...         ],
    ...         Column.COOKING_ACTIVITY_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.END_TIME: [30.0, 150.0, 428.0, 575.0, 705.0, 836.0, 47.0, 215.0, 565.0, 747.0],
    ...         Column.PERSON: ["P03", "P03", "P03", "P03", "P03", "P03", "P54", "P54", "P54", "P54"],
    ...         Column.PERSON_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
    ...         Column.START_TIME_DIFF: [0.0, 30.0, 120.0, 278.0, 147.0, 130.0, 0.0, 47.0, 168.0, 350.0],
    ...     }
    ... )
    >>> groups = group_by_sequence(frame)
    >>> with pl.Config(tbl_cols=-1):
    ...     groups
    shape: (2, 10)
    ┌─────────┬─────────┬─────────┬─────────┬─────────┬────────┬─────────┬─────────┬─────────┬─────────┐
    │ action  ┆ action_ ┆ cooking ┆ cooking ┆ end_tim ┆ person ┆ person_ ┆ sequenc ┆ start_t ┆ start_t │
    │ ---     ┆ id      ┆ _activi ┆ _activi ┆ e       ┆ ---    ┆ id      ┆ e_lengt ┆ ime     ┆ ime_dif │
    │ list[st ┆ ---     ┆ ty      ┆ ty_id   ┆ ---     ┆ str    ┆ ---     ┆ h       ┆ ---     ┆ f       │
    │ r]      ┆ list[i6 ┆ ---     ┆ ---     ┆ list[f6 ┆        ┆ i64     ┆ ---     ┆ list[f6 ┆ ---     │
    │         ┆ 4]      ┆ str     ┆ i64     ┆ 4]      ┆        ┆         ┆ i64     ┆ 4]      ┆ list[f6 │
    │         ┆         ┆         ┆         ┆         ┆        ┆         ┆         ┆         ┆ 4]      │
    ╞═════════╪═════════╪═════════╪═════════╪═════════╪════════╪═════════╪═════════╪═════════╪═════════╡
    │ ["SIL", ┆ [0, 2,  ┆ cereals ┆ 0       ┆ [30.0,  ┆ P03    ┆ 0       ┆ 6       ┆ [1.0,   ┆ [0.0,   │
    │ "take_b ┆ … 0]    ┆         ┆         ┆ 150.0,  ┆        ┆         ┆         ┆ 31.0, … ┆ 30.0, … │
    │ owl", … ┆         ┆         ┆         ┆ …       ┆        ┆         ┆         ┆ 706.0]  ┆ 130.0]  │
    │ "SIL"]  ┆         ┆         ┆         ┆ 836.0]  ┆        ┆         ┆         ┆         ┆         │
    │ ["SIL", ┆ [0, 1,  ┆ milk    ┆ 1       ┆ [47.0,  ┆ P54    ┆ 1       ┆ 4       ┆ [1.0,   ┆ [0.0,   │
    │ "pour_m ┆ … 0]    ┆         ┆         ┆ 215.0,  ┆        ┆         ┆         ┆ 48.0, … ┆ 47.0, … │
    │ ilk", … ┆         ┆         ┆         ┆ …       ┆        ┆         ┆         ┆ 566.0]  ┆ 350.0]  │
    │ "SIL"]  ┆         ┆         ┆         ┆ 747.0]  ┆        ┆         ┆         ┆         ┆         │
    └─────────┴─────────┴─────────┴─────────┴─────────┴────────┴─────────┴─────────┴─────────┴─────────┘

    ```
    """
    data = frame.group_by([Column.PERSON_ID, Column.COOKING_ACTIVITY_ID]).agg(
        pl.first(Column.COOKING_ACTIVITY),
        pl.first(Column.PERSON),
        pl.col(Column.ACTION),
        pl.col(Column.ACTION_ID),
        pl.col(Column.START_TIME),
        pl.col(Column.START_TIME_DIFF),
        pl.col(Column.END_TIME),
        pl.len().cast(pl.Int64).alias(Column.SEQUENCE_LENGTH),
    )
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.PERSON_ID, Column.COOKING_ACTIVITY_ID]),
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
    >>> from arctix.dataset.breakfast import Column, to_array
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION: [
    ...             "SIL",
    ...             "take_bowl",
    ...             "pour_cereals",
    ...             "pour_milk",
    ...             "stir_cereals",
    ...             "SIL",
    ...             "SIL",
    ...             "pour_milk",
    ...             "spoon_powder",
    ...             "SIL",
    ...         ],
    ...         Column.ACTION_ID: [0, 2, 5, 1, 3, 0, 0, 1, 4, 0],
    ...         Column.COOKING_ACTIVITY: [
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...         ],
    ...         Column.COOKING_ACTIVITY_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.END_TIME: [30.0, 150.0, 428.0, 575.0, 705.0, 836.0, 47.0, 215.0, 565.0, 747.0],
    ...         Column.PERSON: ["P03", "P03", "P03", "P03", "P03", "P03", "P54", "P54", "P54", "P54"],
    ...         Column.PERSON_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
    ...         Column.START_TIME_DIFF: [0.0, 30.0, 120.0, 278.0, 147.0, 130.0, 0.0, 47.0, 168.0, 350.0],
    ...     }
    ... )
    >>> arrays = to_array(frame)
    >>> arrays
    {'action': masked_array(
      data=[['SIL', 'take_bowl', 'pour_cereals', 'pour_milk', 'stir_cereals',
             'SIL'],
            ['SIL', 'pour_milk', 'spoon_powder', 'SIL', --, --]],
      mask=[[False, False, False, False, False, False],
            [False, False, False, False,  True,  True]],
      fill_value='N/A',
      dtype='<U12'),
      'action_id': masked_array(
      data=[[0, 2, 5, 1, 3, 0],
            [0, 1, 4, 0, --, --]],
      mask=[[False, False, False, False, False, False],
            [False, False, False, False,  True,  True]],
      fill_value=999999),
      'cooking_activity': array(['cereals', 'milk'], dtype='<U7'),
      'cooking_activity_id': array([0, 1]), 'person': array(['P03', 'P54'], dtype='<U3'),
      'person_id': array([0, 1]),
      'sequence_length': array([6, 4]),
      'start_time': masked_array(
      data=[[1.0, 31.0, 151.0, 429.0, 576.0, 706.0],
            [1.0, 48.0, 216.0, 566.0, --, --]],
      mask=[[False, False, False, False, False, False],
            [False, False, False, False,  True,  True]],
      fill_value=1e+20), 'start_time_diff': masked_array(
      data=[[0.0, 30.0, 120.0, 278.0, 147.0, 130.0],
            [0.0, 47.0, 168.0, 350.0, --, --]],
      mask=[[False, False, False, False, False, False],
            [False, False, False, False,  True,  True]],
      fill_value=1e+20), 'end_time': masked_array(
      data=[[30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
            [47.0, 215.0, 565.0, 747.0, --, --]],
      mask=[[False, False, False, False, False, False],
            [False, False, False, False,  True,  True]],
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
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.COOKING_ACTIVITY: groups.get_column(Column.COOKING_ACTIVITY).to_numpy().astype(str),
        Column.COOKING_ACTIVITY_ID: groups.get_column(Column.COOKING_ACTIVITY_ID)
        .to_numpy()
        .astype(np.int64),
        Column.PERSON: groups.get_column(Column.PERSON).to_numpy().astype(str),
        Column.PERSON_ID: groups.get_column(Column.PERSON_ID).to_numpy().astype(np.int64),
        Column.SEQUENCE_LENGTH: lengths.astype(np.int64),
        Column.START_TIME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.START_TIME_DIFF: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME_DIFF).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.END_TIME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.END_TIME).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
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
    >>> from arctix.dataset.breakfast import Column, to_list
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION: [
    ...             "SIL",
    ...             "take_bowl",
    ...             "pour_cereals",
    ...             "pour_milk",
    ...             "stir_cereals",
    ...             "SIL",
    ...             "SIL",
    ...             "pour_milk",
    ...             "spoon_powder",
    ...             "SIL",
    ...         ],
    ...         Column.ACTION_ID: [0, 2, 5, 1, 3, 0, 0, 1, 4, 0],
    ...         Column.COOKING_ACTIVITY: [
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "cereals",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...             "milk",
    ...         ],
    ...         Column.COOKING_ACTIVITY_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.END_TIME: [30.0, 150.0, 428.0, 575.0, 705.0, 836.0, 47.0, 215.0, 565.0, 747.0],
    ...         Column.PERSON: ["P03", "P03", "P03", "P03", "P03", "P03", "P54", "P54", "P54", "P54"],
    ...         Column.PERSON_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ...         Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
    ...         Column.START_TIME_DIFF: [0.0, 30.0, 120.0, 278.0, 147.0, 130.0, 0.0, 47.0, 168.0, 350.0],
    ...     }
    ... )
    >>> data_list = to_list(frame)
    >>> data_list
    {'action': [['SIL', 'take_bowl', 'pour_cereals', 'pour_milk', 'stir_cereals', 'SIL'], ['SIL', 'pour_milk', 'spoon_powder', 'SIL']],
     'action_id': [[0, 2, 5, 1, 3, 0], [0, 1, 4, 0]],
     'cooking_activity': ['cereals', 'milk'],
     'cooking_activity_id': [0, 1],
     'end_time': [[30.0, 150.0, 428.0, 575.0, 705.0, 836.0], [47.0, 215.0, 565.0, 747.0]],
     'person': ['P03', 'P54'],
     'person_id': [0, 1],
     'sequence_length': [6, 4],
     'start_time': [[1.0, 31.0, 151.0, 429.0, 576.0, 706.0], [1.0, 48.0, 216.0, 566.0]],
     'start_time_diff': [[0.0, 30.0, 120.0, 278.0, 147.0, 130.0], [0.0, 47.0, 168.0, 350.0]]}

    ```
    """
    return group_by_sequence(frame).to_dict(as_series=False)


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("breakfast")
    raw_data = fetch_data(path, name="segmentation_coarse")
    logger.info(f"data_raw:\n{raw_data}")
    data, metadata = prepare_data(raw_data)
    logger.info(f"data:\n{data}")
    logger.info(f"metadata:\n{metadata}")
