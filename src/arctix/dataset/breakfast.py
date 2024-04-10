r"""Contain code to download and prepare the Breakfast data.

Information about the Breakfast dataset can be found in the following
paper:

The Language of Actions: Recovering the Syntax and Semantics of Goal-
Directed Human Activities. Kuehne, Arslan, and Serre. CVPR 2014.

Project page:

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

Data can be downloaded at

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/#Downloads

The documentation assumes the data are downloaded in the directory `/path/to/data/breakfast/`.
"""

from __future__ import annotations

__all__ = [
    "download_data",
    "fetch_data",
    "load_annotation_file",
    "load_data",
    "parse_annotation_lines",
    "prepare_data",
]

import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from iden.utils.path import sanitize_path

from arctix.transformer import dataframe as td
from arctix.utils.dataframe import drop_duplicates, generate_vocabulary
from arctix.utils.download import download_drive_file
from arctix.utils.iter import FileFilter, PathLister
from arctix.utils.mapping import convert_to_dict_of_flat_lists

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


class Column:
    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    COOKING_ACTIVITY: str = "cooking_activity"
    END_TIME: str = "end_time"
    PERSON: str = "person"
    PERSON_ID: str = "person_id"
    START_TIME: str = "start_time"


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
    """
    paths = FileFilter(PathLister([sanitize_path(path)], pattern="**/*.txt"))
    annotations = list(map(load_annotation_file, paths))
    data = convert_to_dict_of_flat_lists(annotations)
    data = pl.DataFrame(data)
    if remove_duplicate:
        data = drop_duplicates(data)
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


def prepare_data(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    r"""Prepare the data.

    Args:
        frame: The raw DataFrame.

    Returns:
        A tuple containing the prepared data and the metadata.
    """
    vocab_action = generate_vocabulary(frame, col=Column.ACTION)
    vocab_person = generate_vocabulary(frame, col=Column.PERSON)
    transformer = td.Sequential(
        [
            td.TokenToIndex(
                vocab=vocab_action, token_column=Column.ACTION, index_column=Column.ACTION_ID
            ),
            td.TokenToIndex(
                vocab=vocab_person, token_column=Column.PERSON, index_column=Column.PERSON_ID
            ),
        ]
    )
    out = transformer.transform(frame)
    return out, {"vocab_action": vocab_action, "vocab_person": vocab_person}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)

    path = Path("~/Downloads/breakfast")
    download_data(path)
    data = load_data(path.joinpath("segmentation_coarse"))
    logger.info(f"data:\n{data}")
