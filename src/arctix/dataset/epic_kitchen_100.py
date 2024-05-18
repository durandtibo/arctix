r"""Contain code to download and prepare the EPIC-KITCHENS-100 data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/epic_kitchen_100/`.
"""

from __future__ import annotations

__all__ = [
    "ANNOTATION_FILENAMES",
    "ANNOTATION_URL",
    "Column",
    "MetadataKeys",
    "NUM_NOUNS",
    "NUM_VERBS",
    "download_data",
    "fetch_data",
    "group_by_sequence",
    "is_annotation_path_ready",
    "load_data",
    "load_event_data",
    "load_noun_vocab",
    "load_verb_vocab",
    "prepare_data",
    "to_array",
    "to_list",
]

import logging
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from iden.utils.path import sanitize_path

from arctix.transformer import dataframe as td
from arctix.utils.download import download_url_to_file
from arctix.utils.masking import convert_sequences_to_array, generate_mask_from_lengths
from arctix.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)

ANNOTATION_URL = (
    "https://github.com/epic-kitchens/epic-kitchens-100-annotations/archive/refs/heads/master.zip"
)

ANNOTATION_FILENAMES = [
    "EPIC_100_noun_classes.csv",
    "EPIC_100_tail_nouns.csv",
    "EPIC_100_tail_verbs.csv",
    "EPIC_100_test_timestamps.csv",
    "EPIC_100_train.csv",
    "EPIC_100_unseen_participant_ids_test.csv",
    "EPIC_100_unseen_participant_ids_validation.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_verb_classes.csv",
    "EPIC_100_video_info.csv",
]

NUM_NOUNS = 300
NUM_VERBS = 97


class Column:
    r"""Indicate the column names."""

    ALL_NOUNS: str = "all_nouns"
    ALL_NOUN_IDS: str = "all_noun_classes"
    NARRATION: str = "narration"
    NARRATION_ID: str = "narration_id"
    NARRATION_TIMESTAMP: str = "narration_timestamp"
    NOUN: str = "noun"
    NOUN_ID: str = "noun_class"
    PARTICIPANT_ID: str = "participant_id"
    SEQUENCE_LENGTH: str = "sequence_length"
    START_FRAME: str = "start_frame"
    START_TIMESTAMP: str = "start_timestamp"
    START_TIME_SECOND: str = "start_time_second"
    START_TIME_SECOND_DIFF: str = "start_time_second_diff"
    STOP_FRAME: str = "stop_frame"
    STOP_TIMESTAMP: str = "stop_timestamp"
    STOP_TIME_SECOND: str = "stop_time_second"
    VERB: str = "verb"
    VERB_ID: str = "verb_class"
    VIDEO_ID: str = "video_id"


class MetadataKeys:
    r"""Indicate the metadata keys."""

    VOCAB_NOUN: str = "vocab_noun"
    VOCAB_VERB: str = "vocab_verb"


def fetch_data(path: Path, split: str, force_download: bool = False) -> tuple[pl.DataFrame, dict]:
    r"""Download and load the data for EPIC-KITCHENS-100 dataset.

    Args:
        path: The path where to store the downloaded data.
        split: The dataset split.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Returns:
        The annotations in a DataFrame and the metadata.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.multithumos import fetch_data
    >>> data, metadata = fetch_data(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    download_data(path, force_download)
    return load_data(path, split)


def download_data(path: Path, force_download: bool = False) -> None:
    r"""Download the EPIC-KITCHENS-100 annotations.

    Args:
        path: The path where to store the downloaded data.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import download_data
    >>> download_data(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    if not is_annotation_path_ready(path) or force_download:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_file = tmp_path.joinpath("epic-kitchens-100.zip.partial")
            logger.info(f"downloading EPIC-KITCHENS-100 annotations data in {zip_file}...")
            download_url_to_file(ANNOTATION_URL, zip_file.as_posix(), progress=True)

            logger.info(f"extracting {zip_file} in {tmp_path}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmp_path)

            logger.info(f"moving extracted files to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            for filename in ANNOTATION_FILENAMES:
                tmp_path.joinpath(f"epic-kitchens-100-annotations-master/{filename}").rename(
                    path.joinpath(filename)
                )

    logger.info(f"EPIC-KITCHENS-100 annotation data are available in {path}")


def is_annotation_path_ready(path: Path) -> bool:
    r"""Indicate if the given path contains the EPIC-KITCHENS-100
    annotation data.

    Args:
        path: The path to check.

    Returns:
        ``True`` if the path contains the EPIC-KITCHENS-100 data,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import is_annotation_path_ready
    >>> is_annotation_path_ready(Path("/path/to/data/"))
    False

    ```
    """
    path = sanitize_path(path)
    return all(path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def load_data(path: Path, split: str) -> tuple[pl.DataFrame, dict]:
    r"""Load all the annotations in a DataFrame and the metadata.

    Args:
        path: The directory where the dataset annotations are stored.
        split: The dataset split.

    Returns:
        The annotations in a DataFrame and the metadata.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import load_data
    >>> data, metadata = load_data(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    data = load_event_data(path.joinpath(f"EPIC_100_{split}.csv"))
    metadata = {
        MetadataKeys.VOCAB_NOUN: load_noun_vocab(path),
        MetadataKeys.VOCAB_VERB: load_verb_vocab(path),
    }
    return data, metadata


def load_event_data(path: Path) -> pl.DataFrame:
    r"""Load the event data from a CSV file.

    Args:
        path: The path to the CSV file.

    Returns:
        The event data in a ``polars.DataFrame``.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import load_event_data
    >>> data, metadata = load_event_data(Path("/path/to/data/epic_kitchen_100/EPIC_100_train.csv"))  # doctest: +SKIP

    ```
    """
    frame = pl.read_csv(
        path,
        dtypes={
            Column.ALL_NOUNS: pl.String,
            Column.ALL_NOUN_IDS: pl.String,
            Column.NARRATION: pl.String,
            Column.NARRATION_ID: pl.String,
            Column.NARRATION_TIMESTAMP: pl.String,
            Column.NOUN: pl.String,
            Column.NOUN_ID: pl.Int64,
            Column.PARTICIPANT_ID: pl.String,
            Column.START_FRAME: pl.Int64,
            Column.START_TIMESTAMP: pl.String,
            Column.STOP_FRAME: pl.Int64,
            Column.STOP_TIMESTAMP: pl.String,
            Column.VERB: pl.String,
            Column.VERB_ID: pl.Int64,
            Column.VIDEO_ID: pl.String,
        },
    )
    transformer = td.Sequential(
        [
            td.ToTime(
                columns=[Column.START_TIMESTAMP, Column.NARRATION_TIMESTAMP, Column.STOP_TIMESTAMP],
                format="%H:%M:%S%.3f",
            ),
            td.JsonDecode(columns=[Column.ALL_NOUN_IDS], dtype=pl.List(pl.Int64)),
            td.JsonDecode(columns=[Column.ALL_NOUNS], dtype=pl.List(pl.String)),
            td.SortColumns(),
        ]
    )
    data = transformer.transform(frame)
    if data.select(pl.len()).item():
        data = data.sort(by=[Column.VIDEO_ID, Column.START_FRAME])
    return data


def load_noun_vocab(path: Path) -> Vocabulary:
    r"""Load the vocabulary of nouns.

    Args:
        path: The directory where the dataset annotations are stored.
            The directory must contain the
            ``EPIC_100_noun_classes.csv`` file.

    Returns:
        The vocabulary for nouns.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import load_noun_vocab
    >>> noun_vocab = load_noun_vocab(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    path = path.joinpath("EPIC_100_noun_classes.csv")
    logger.info(f"loading noun vocabulary from {path}...")
    frame = pl.read_csv(path, columns=["id", "key"], dtypes={"id": pl.Int64, "key": pl.String})
    vocab = Vocabulary.from_token_to_index(
        {token: i for i, token in zip(frame["id"], frame["key"])}
    )
    if (count := len(vocab)) != NUM_NOUNS:
        msg = f"Expected {NUM_NOUNS} nouns but received {count:,}"
        raise RuntimeError(msg)
    return vocab


def load_verb_vocab(path: Path) -> Vocabulary:
    r"""Load the vocabulary of verbs.

    Args:
        path: The directory where the dataset annotations are stored.
            The directory must contain the
            ``EPIC_100_verb_classes.csv`` file.

    Returns:
        The vocabulary for verbs.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import load_verb_vocab
    >>> verb_vocab = load_verb_vocab(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    path = path.joinpath("EPIC_100_verb_classes.csv")
    logger.info(f"loading verb vocabulary from {path}...")
    frame = pl.read_csv(path, columns=["id", "key"], dtypes={"id": pl.Int64, "key": pl.String})
    vocab = Vocabulary.from_token_to_index(
        {token: i for i, token in zip(frame["id"], frame["key"])}
    )
    if (count := len(vocab)) != NUM_VERBS:
        msg = f"Expected {NUM_VERBS} verbs but received {count:,}"
        raise RuntimeError(msg)
    return vocab


def prepare_data(frame: pl.DataFrame, metadata: dict) -> tuple[pl.DataFrame, dict]:
    r"""Prepare the data.

    Args:
        frame: The raw DataFrame.
        metadata: The metadata wich contains the vocabularies to
            convert verbs and nouns to index.

    Returns:
        A tuple containing the prepared data and the metadata.

    Example usage:

    ```pycon

    >>> import datetime
    >>> import polars as pl
    >>> from arctix.dataset.epic_kitchen_100 import Column, prepare_data
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ALL_NOUN_IDS: [[3], [114], [3]],
    ...         Column.ALL_NOUNS: [["door"], ["light"], ["door"]],
    ...         Column.NARRATION: ["open door", "turn on light", "close door"],
    ...         Column.NARRATION_ID: ["P01_01_0", "P01_01_1", "P01_01_2"],
    ...         Column.NARRATION_TIMESTAMP: [
    ...             datetime.time(0, 0, 1, 89000),
    ...             datetime.time(0, 0, 2, 629000),
    ...             datetime.time(0, 0, 5, 349000),
    ...         ],
    ...         Column.NOUN: ["door", "light", "door"],
    ...         Column.NOUN_ID: [3, 114, 3],
    ...         Column.PARTICIPANT_ID: ["P01", "P01", "P01"],
    ...         Column.START_FRAME: [8, 262, 418],
    ...         Column.START_TIMESTAMP: [
    ...             datetime.time(0, 0, 0, 140000),
    ...             datetime.time(0, 0, 4, 370000),
    ...             datetime.time(0, 0, 6, 980000),
    ...         ],
    ...         Column.STOP_FRAME: [202, 370, 569],
    ...         Column.STOP_TIMESTAMP: [
    ...             datetime.time(0, 0, 3, 370000),
    ...             datetime.time(0, 0, 6, 170000),
    ...             datetime.time(0, 0, 9, 490000),
    ...         ],
    ...         Column.VERB: ["open", "turn-on", "close"],
    ...         Column.VERB_ID: [3, 6, 4],
    ...         Column.VIDEO_ID: ["P01_01", "P01_01", "P01_01"],
    ...     },
    ...     schema={
    ...         Column.ALL_NOUN_IDS: pl.List(pl.Int64),
    ...         Column.ALL_NOUNS: pl.List(pl.String),
    ...         Column.NARRATION: pl.String,
    ...         Column.NARRATION_ID: pl.String,
    ...         Column.NARRATION_TIMESTAMP: pl.Time,
    ...         Column.NOUN: pl.String,
    ...         Column.NOUN_ID: pl.Int64,
    ...         Column.PARTICIPANT_ID: pl.String,
    ...         Column.START_FRAME: pl.Int64,
    ...         Column.START_TIMESTAMP: pl.Time,
    ...         Column.STOP_FRAME: pl.Int64,
    ...         Column.STOP_TIMESTAMP: pl.Time,
    ...         Column.VERB: pl.String,
    ...         Column.VERB_ID: pl.Int64,
    ...         Column.VIDEO_ID: pl.String,
    ...     },
    ... )
    >>> data, metadata = prepare_data(frame, metadata={})
    >>> with pl.Config(tbl_cols=-1):
    ...     data
    shape: (3, 18)
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ all ┆ all ┆ nar ┆ nar ┆ nar ┆ nou ┆ nou ┆ par ┆ sta ┆ sta ┆ sta ┆ sta ┆ sto ┆ sto ┆ sto ┆ ver ┆ ver ┆ vid │
    │ _no ┆ _no ┆ rat ┆ rat ┆ rat ┆ n   ┆ n_c ┆ tic ┆ rt_ ┆ rt_ ┆ rt_ ┆ rt_ ┆ p_f ┆ p_t ┆ p_t ┆ b   ┆ b_c ┆ eo_ │
    │ un_ ┆ uns ┆ ion ┆ ion ┆ ion ┆ --- ┆ las ┆ ipa ┆ fra ┆ tim ┆ tim ┆ tim ┆ ram ┆ ime ┆ ime ┆ --- ┆ las ┆ id  │
    │ cla ┆ --- ┆ --- ┆ _id ┆ _ti ┆ str ┆ s   ┆ nt_ ┆ me  ┆ e_s ┆ e_s ┆ est ┆ e   ┆ _se ┆ sta ┆ str ┆ s   ┆ --- │
    │ sse ┆ lis ┆ str ┆ --- ┆ mes ┆     ┆ --- ┆ id  ┆ --- ┆ eco ┆ eco ┆ amp ┆ --- ┆ con ┆ mp  ┆     ┆ --- ┆ str │
    │ s   ┆ t[s ┆     ┆ str ┆ tam ┆     ┆ i64 ┆ --- ┆ i64 ┆ nd  ┆ nd_ ┆ --- ┆ i64 ┆ d   ┆ --- ┆     ┆ i64 ┆     │
    │ --- ┆ tr] ┆     ┆     ┆ p   ┆     ┆     ┆ str ┆     ┆ --- ┆ dif ┆ tim ┆     ┆ --- ┆ tim ┆     ┆     ┆     │
    │ lis ┆     ┆     ┆     ┆ --- ┆     ┆     ┆     ┆     ┆ f64 ┆ f   ┆ e   ┆     ┆ f64 ┆ e   ┆     ┆     ┆     │
    │ t[i ┆     ┆     ┆     ┆ tim ┆     ┆     ┆     ┆     ┆     ┆ --- ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
    │ 64] ┆     ┆     ┆     ┆ e   ┆     ┆     ┆     ┆     ┆     ┆ f64 ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
    ╞═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╡
    │ [3] ┆ ["d ┆ ope ┆ P01 ┆ 00: ┆ doo ┆ 3   ┆ P01 ┆ 8   ┆ 0.1 ┆ 0.0 ┆ 00: ┆ 202 ┆ 3.3 ┆ 00: ┆ ope ┆ 3   ┆ P01 │
    │     ┆ oor ┆ n   ┆ _01 ┆ 00: ┆ r   ┆     ┆     ┆     ┆ 4   ┆     ┆ 00: ┆     ┆ 7   ┆ 00: ┆ n   ┆     ┆ _01 │
    │     ┆ "]  ┆ doo ┆ _0  ┆ 01. ┆     ┆     ┆     ┆     ┆     ┆     ┆ 00. ┆     ┆     ┆ 03. ┆     ┆     ┆     │
    │     ┆     ┆ r   ┆     ┆ 089 ┆     ┆     ┆     ┆     ┆     ┆     ┆ 140 ┆     ┆     ┆ 370 ┆     ┆     ┆     │
    │ [11 ┆ ["l ┆ tur ┆ P01 ┆ 00: ┆ lig ┆ 114 ┆ P01 ┆ 262 ┆ 4.3 ┆ 4.2 ┆ 00: ┆ 370 ┆ 6.1 ┆ 00: ┆ tur ┆ 6   ┆ P01 │
    │ 4]  ┆ igh ┆ n   ┆ _01 ┆ 00: ┆ ht  ┆     ┆     ┆     ┆ 7   ┆ 3   ┆ 00: ┆     ┆ 7   ┆ 00: ┆ n-o ┆     ┆ _01 │
    │     ┆ t"] ┆ on  ┆ _1  ┆ 02. ┆     ┆     ┆     ┆     ┆     ┆     ┆ 04. ┆     ┆     ┆ 06. ┆ n   ┆     ┆     │
    │     ┆     ┆ lig ┆     ┆ 629 ┆     ┆     ┆     ┆     ┆     ┆     ┆ 370 ┆     ┆     ┆ 170 ┆     ┆     ┆     │
    │     ┆     ┆ ht  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
    │ [3] ┆ ["d ┆ clo ┆ P01 ┆ 00: ┆ doo ┆ 3   ┆ P01 ┆ 418 ┆ 6.9 ┆ 2.6 ┆ 00: ┆ 569 ┆ 9.4 ┆ 00: ┆ clo ┆ 4   ┆ P01 │
    │     ┆ oor ┆ se  ┆ _01 ┆ 00: ┆ r   ┆     ┆     ┆     ┆ 8   ┆ 1   ┆ 00: ┆     ┆ 9   ┆ 00: ┆ se  ┆     ┆ _01 │
    │     ┆ "]  ┆ doo ┆ _2  ┆ 05. ┆     ┆     ┆     ┆     ┆     ┆     ┆ 06. ┆     ┆     ┆ 09. ┆     ┆     ┆     │
    │     ┆     ┆ r   ┆     ┆ 349 ┆     ┆     ┆     ┆     ┆     ┆     ┆ 980 ┆     ┆     ┆ 490 ┆     ┆     ┆     │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
    >>> metadata
    {}

    ```
    """
    transformer = td.Sequential(
        [
            td.TimeToSecond(in_col=Column.START_TIMESTAMP, out_col=Column.START_TIME_SECOND),
            td.TimeToSecond(in_col=Column.STOP_TIMESTAMP, out_col=Column.STOP_TIME_SECOND),
            td.TimeDiff(
                group_cols=[Column.VIDEO_ID],
                time_col=Column.START_TIME_SECOND,
                time_diff_col=Column.START_TIME_SECOND_DIFF,
            ),
            td.Sort(columns=[Column.VIDEO_ID, Column.START_FRAME]),
            td.Cast(columns=[Column.START_TIME_SECOND, Column.STOP_TIME_SECOND], dtype=pl.Float64),
            td.SortColumns(),
        ]
    )
    out = transformer.transform(frame)
    return out, metadata


def group_by_sequence(frame: pl.DataFrame) -> pl.DataFrame:
    r"""Group the DataFrame by sequences of actions.

    Args:
        frame: The input DataFrame.

    Returns:
        The DataFrame after the grouping.
    """
    data = (
        frame.sort(by=[Column.VIDEO_ID, Column.START_FRAME])
        .group_by([Column.VIDEO_ID])
        .agg(
            pl.first(Column.PARTICIPANT_ID),
            pl.col(Column.ALL_NOUNS),
            pl.col(Column.ALL_NOUN_IDS),
            pl.col(Column.NARRATION),
            pl.col(Column.NARRATION_ID),
            pl.col(Column.NARRATION_TIMESTAMP),
            pl.col(Column.NOUN),
            pl.col(Column.NOUN_ID),
            pl.col(Column.START_FRAME),
            pl.col(Column.START_TIMESTAMP),
            pl.col(Column.START_TIME_SECOND),
            pl.col(Column.START_TIME_SECOND_DIFF),
            pl.col(Column.STOP_FRAME),
            pl.col(Column.STOP_TIMESTAMP),
            pl.col(Column.STOP_TIME_SECOND),
            pl.col(Column.VERB),
            pl.col(Column.VERB_ID),
            pl.len().cast(pl.Int64).alias(Column.SEQUENCE_LENGTH),
        )
    )
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.VIDEO_ID]),
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
    """
    groups = group_by_sequence(frame)
    lengths = groups.get_column(Column.SEQUENCE_LENGTH).to_numpy()
    mask = generate_mask_from_lengths(lengths)
    return {
        Column.NARRATION: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.NARRATION).to_list(),
                max_len=mask.shape[1],
                dtype=np.object_,
                padded_value="N/A",
            ).astype(str),
            mask=mask,
        ),
        Column.NARRATION_ID: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.NARRATION_ID).to_list(),
                max_len=mask.shape[1],
                dtype=np.object_,
                padded_value="N/A",
            ).astype(str),
            mask=mask,
        ),
        Column.NOUN: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.NOUN).to_list(),
                max_len=mask.shape[1],
                dtype=np.object_,
                padded_value="N/A",
            ).astype(str),
            mask=mask,
        ),
        Column.NOUN_ID: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.NOUN_ID).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.PARTICIPANT_ID: groups.get_column(Column.PARTICIPANT_ID).to_numpy().astype(str),
        Column.SEQUENCE_LENGTH: groups.get_column(Column.SEQUENCE_LENGTH)
        .to_numpy()
        .astype(np.int64),
        Column.START_FRAME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_FRAME).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.START_TIME_SECOND: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME_SECOND).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.START_TIME_SECOND_DIFF: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.START_TIME_SECOND_DIFF).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.STOP_FRAME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.STOP_FRAME).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.STOP_TIME_SECOND: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.STOP_TIME_SECOND).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.VERB: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.VERB).to_list(),
                max_len=mask.shape[1],
                dtype=np.object_,
                padded_value="N/A",
            ).astype(str),
            mask=mask,
        ),
        Column.VERB_ID: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.VERB_ID).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.VIDEO_ID: groups.get_column(Column.VIDEO_ID).to_numpy().astype(str),
    }


def to_list(frame: pl.DataFrame) -> dict[str, list]:
    r"""Convert a DataFrame to a dictionary of lists.

    Args:
        frame: The input DataFrame.

    Returns:
        The dictionary of lists.
    """
    return group_by_sequence(frame).to_dict(as_series=False)


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("epic_kitchen_100")
    data_raw, metadata_raw = fetch_data(path, split="train")
    logger.info(f"data_raw:\n{data_raw}")
    logger.info(f"metadata_raw:\n{metadata_raw}")
    data, metadata = prepare_data(data_raw, metadata_raw)
    logger.info(f"data:\n{data}")
    logger.info(f"metadata:\n{metadata}")

    arrays = to_array(data)
    logger.info(f"arrays:\n{arrays}")
