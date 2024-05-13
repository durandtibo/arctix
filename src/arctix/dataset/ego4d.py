r"""Contain code to prepare the Ego4D data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/ego4d/`.
"""

from __future__ import annotations

__all__ = [
    "Column",
    "MetadataKeys",
    "NUM_NOUNS",
    "NUM_VERBS",
    "load_event_data",
    "load_data",
    "load_noun_vocab",
    "load_taxonomy_vocab",
    "load_verb_vocab",
    "prepare_data",
]

import logging
from collections import Counter
from pathlib import Path

import polars as pl
from iden.io import load_json

from arctix.transformer import dataframe as td
from arctix.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)

NUM_NOUNS = 521
NUM_VERBS = 117


class Column:
    r"""Indicate the column names."""

    ACTION_END_FRAME: str = "action_clip_end_frame"
    ACTION_END_SEC: str = "action_clip_end_sec"
    ACTION_INDEX: str = "action_idx"
    ACTION_START_FRAME: str = "action_clip_start_frame"
    ACTION_START_SEC: str = "action_clip_start_sec"
    CLIP_ID: str = "clip_uid"
    NOUN: str = "noun"
    NOUN_ID: str = "noun_label"
    VERB: str = "verb"
    VERB_ID: str = "verb_label"
    VIDEO_ID: str = "video_uid"
    SPLIT: str = "split"
    SEQUENCE_LENGTH: str = "sequence_length"


class MetadataKeys:
    r"""Indicate the metadata keys."""

    VOCAB_NOUN: str = "vocab_noun"
    VOCAB_VERB: str = "vocab_verb"


def fetch_data(path: Path, split: str) -> tuple[pl.DataFrame, dict]:
    r"""Download and load the data and the metadata.

    Notes:
        This function does not implement the data downloading because
        it is necessary to get credentials to access the data.

    Args:
        path: The directory where the dataset annotations are stored.
        split: The dataset split.

    Returns:
        The annotations in a DataFrame and the metadata.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import fetch_data
    >>> data, metadata = fetch_data(
    ...     Path("/path/to/data/ego4d/"), split="train"
    ... )  # doctest: +SKIP

    ```
    """
    return load_data(path=path, split=split)


def load_data(path: Path, split: str) -> tuple[pl.DataFrame, dict]:
    r"""Load the data and the metadata.

    Args:
        path: The directory where the dataset annotations are stored.
        split: The dataset split.

    Returns:
        The annotations in a DataFrame and the metadata.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import load_data
    >>> data, metadata = load_data(
    ...     Path("/path/to/data/ego4d/"), split="train"
    ... )  # doctest: +SKIP

    ```
    """
    data = load_event_data(path=path, split=split)
    metadata = {
        MetadataKeys.VOCAB_NOUN: load_noun_vocab(path),
        MetadataKeys.VOCAB_VERB: load_verb_vocab(path),
    }
    return data, metadata


def load_event_data(path: Path, split: str) -> pl.DataFrame:
    r"""Load the data in the annotation file into a DataFrame.

    Args:
        path: The path to the Ego4D annotations data.
        split: The dataset split to load.
            Expect ``'train'`` or ``'val'``.

    Returns:
        The annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import load_event_data
    >>> data = load_event_data(
    ...     Path("/path/to/data/ego4d/"), split="train"
    ... )  # doctest: +SKIP

    ```
    """
    path = path.joinpath(f"ego4d_data/v2/annotations/fho_lta_{split}.json")
    logger.info(f"loading data for {split} split...")
    data = load_json(path)
    frame = (
        pl.DataFrame(
            data["clips"],
            schema={
                Column.ACTION_END_FRAME: pl.Int64,
                Column.ACTION_END_SEC: pl.Float64,
                Column.ACTION_START_FRAME: pl.Int64,
                Column.ACTION_START_SEC: pl.Float64,
                Column.ACTION_INDEX: pl.Int64,
                Column.CLIP_ID: pl.String,
                Column.NOUN: pl.String,
                Column.NOUN_ID: pl.Int64,
                Column.VERB: pl.String,
                Column.VERB_ID: pl.Int64,
                Column.VIDEO_ID: pl.String,
                Column.SPLIT: pl.String,
            },
        )
        .select(
            [
                Column.ACTION_END_FRAME,
                Column.ACTION_END_SEC,
                Column.ACTION_START_FRAME,
                Column.ACTION_START_SEC,
                Column.ACTION_INDEX,
                Column.CLIP_ID,
                Column.NOUN,
                Column.NOUN_ID,
                Column.VERB,
                Column.VERB_ID,
                Column.VIDEO_ID,
            ]
        )
        .with_columns(pl.lit(split).alias(Column.SPLIT))
    )
    transformer = td.Sequential(
        [
            td.Sort(columns=[Column.VIDEO_ID, Column.CLIP_ID, Column.ACTION_INDEX]),
            td.SortColumns(),
        ]
    )
    return transformer.transform(frame)


def load_noun_vocab(path: Path) -> Vocabulary:
    r"""Load the vocabulary of nouns.

    Args:
        path: The path to the Ego4D annotations data.

    Returns:
        The vocabulary for nouns.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import load_noun_vocab
    >>> vocab_noun = load_noun_vocab(Path("/path/to/data/ego4d/"))  # doctest: +SKIP

    ```
    """
    return load_taxonomy_vocab(path, name="nouns", expected_size=NUM_NOUNS)


def load_verb_vocab(path: Path) -> Vocabulary:
    r"""Load the vocabulary of verbs.

    Args:
        path: The path to the Ego4D annotations data.

    Returns:
        The vocabulary for verbs.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import load_verb_vocab
    >>> vocab_verb = load_verb_vocab(Path("/path/to/data/ego4d/"))  # doctest: +SKIP

    ```
    """
    return load_taxonomy_vocab(path, name="verbs", expected_size=NUM_VERBS)


def load_taxonomy_vocab(path: Path, name: str, expected_size: int | None = None) -> Vocabulary:
    r"""Load a vocabulary from the taxonomy annotation file.

    Args:
        path: The path to the Ego4D annotations data.
        name: The taxonomy name to load. The valid values are
            ``'nouns'`` and ``'verbs'``.
        expected_size: Indicate the expected vocabulary size.
            If ``None``, the size is not checked.

    Returns:
        The vocabulary associated to the given taxonomy.

    Raises:
        RuntimeError: if the name is incorrect.
        RuntimeError: if the vocabulary size is incorrect.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.ego4d import load_taxonomy_vocab
    >>> data = load_taxonomy_vocab(
    ...     Path("/path/to/data/ego4d/"), name="nouns"
    ... )  # doctest: +SKIP

    ```
    """
    if name not in {"nouns", "verbs"}:
        msg = f"Incorrect taxonomy name: {name}. The valid values are 'nouns' and 'verbs'"
        raise RuntimeError(msg)
    path = path.joinpath("ego4d_data/v2/annotations/fho_lta_taxonomy.json")
    logger.info(f"loading taxonomy data from {path}...")
    data = load_json(path)
    vocab = Vocabulary(Counter({token: 1 for token in data[name]}))
    if expected_size is not None and (count := len(vocab)) != expected_size:
        msg = f"Expected {expected_size} {name} but received {count:,}"
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

    >>> import polars as pl
    >>> from arctix.dataset.ego4d import Column, prepare_data
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
    ...         Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
    ...         Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
    ...         Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
    ...         Column.ACTION_INDEX: [0, 1, 2, 0, 1],
    ...         Column.CLIP_ID: ["clip1", "clip1", "clip1", "clip2", "clip2"],
    ...         Column.NOUN: ["noun2", "noun3", "noun1", "noun1", "noun2"],
    ...         Column.NOUN_ID: [2, 3, 1, 1, 2],
    ...         Column.SPLIT: ["train", "train", "train", "train", "train"],
    ...         Column.VERB: ["verb4", "verb2", "verb1", "verb1", "verb2"],
    ...         Column.VERB_ID: [4, 2, 1, 1, 2],
    ...         Column.VIDEO_ID: ["video1", "video1", "video1", "video2", "video2"],
    ...     }
    ... )
    >>> data, metadata = prepare_data(frame, metadata={})
    >>> with pl.Config(tbl_cols=-1):
    ...     data
    shape: (5, 12)
    ┌────────┬────────┬────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ action ┆ action ┆ action ┆ actio ┆ actio ┆ clip_ ┆ noun  ┆ noun_ ┆ split ┆ verb  ┆ verb_ ┆ video │
    │ _clip_ ┆ _clip_ ┆ _clip_ ┆ n_cli ┆ n_idx ┆ uid   ┆ ---   ┆ label ┆ ---   ┆ ---   ┆ label ┆ _uid  │
    │ end_fr ┆ end_se ┆ start_ ┆ p_sta ┆ ---   ┆ ---   ┆ str   ┆ ---   ┆ str   ┆ str   ┆ ---   ┆ ---   │
    │ ame    ┆ c      ┆ frame  ┆ rt_se ┆ i64   ┆ str   ┆       ┆ i64   ┆       ┆       ┆ i64   ┆ str   │
    │ ---    ┆ ---    ┆ ---    ┆ c     ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ i64    ┆ f64    ┆ i64    ┆ ---   ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │        ┆        ┆        ┆ f64   ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    ╞════════╪════════╪════════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
    │ 47     ┆ 4.7    ┆ 23     ┆ 2.3   ┆ 0     ┆ clip1 ┆ noun2 ┆ 2     ┆ train ┆ verb4 ┆ 4     ┆ video │
    │        ┆        ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 82     ┆ 8.2    ┆ 39     ┆ 3.9   ┆ 1     ┆ clip1 ┆ noun3 ┆ 3     ┆ train ┆ verb2 ┆ 2     ┆ video │
    │        ┆        ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 102    ┆ 10.2   ┆ 74     ┆ 7.4   ┆ 2     ┆ clip1 ┆ noun1 ┆ 1     ┆ train ┆ verb1 ┆ 1     ┆ video │
    │        ┆        ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 74     ┆ 7.4    ┆ 12     ┆ 1.2   ┆ 0     ┆ clip2 ┆ noun1 ┆ 1     ┆ train ┆ verb1 ┆ 1     ┆ video │
    │        ┆        ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 2     │
    │ 142    ┆ 14.2   ┆ 82     ┆ 8.2   ┆ 1     ┆ clip2 ┆ noun2 ┆ 2     ┆ train ┆ verb2 ┆ 2     ┆ video │
    │        ┆        ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 2     │
    └────────┴────────┴────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

    >>> metadata
    {}

    ```
    """
    transformer = td.Sequential(
        [
            td.Cast(
                columns=[
                    Column.ACTION_INDEX,
                    Column.ACTION_END_FRAME,
                    Column.ACTION_START_FRAME,
                    Column.NOUN_ID,
                    Column.VERB_ID,
                ],
                dtype=pl.Int64,
            ),
            td.Cast(columns=[Column.ACTION_START_SEC, Column.ACTION_END_SEC], dtype=pl.Float64),
            td.Sort(columns=[Column.VIDEO_ID, Column.CLIP_ID, Column.ACTION_INDEX]),
            td.SortColumns(),
        ]
    )
    out = transformer.transform(frame)
    return out, metadata


def group_by_sequence(frame: pl.DataFrame, group_col: str = Column.CLIP_ID) -> pl.DataFrame:
    r"""Group the DataFrame by sequences of actions.

    Args:
        frame: The input DataFrame.
        group_col: The column used to generate the sequence.

    Returns:
        The DataFrame after the grouping.
    """
    data = (
        frame.sort(by=[Column.VIDEO_ID, Column.CLIP_ID, Column.ACTION_INDEX])
        .group_by([group_col])
        .agg(
            pl.col(Column.ACTION_END_FRAME),
            pl.col(Column.ACTION_END_SEC),
            pl.col(Column.ACTION_START_FRAME),
            pl.col(Column.ACTION_START_SEC),
            pl.col(Column.NOUN),
            pl.col(Column.NOUN_ID),
            pl.col(Column.VERB),
            pl.col(Column.VERB_ID),
            pl.first(Column.SPLIT),
            pl.len().cast(pl.Int64).alias(Column.SEQUENCE_LENGTH),
        )
    )
    transformer = td.Sequential(
        [
            td.Sort(columns=[group_col]),
            td.SortColumns(),
        ]
    )
    return transformer.transform(data)


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("ego4d")
    data_raw, metadata_raw = fetch_data(path, split="train")
    logger.info(f"data_raw:\n{data_raw}")
    logger.info(f"metadata_raw:\n{metadata_raw}")

    data, metadata = prepare_data(data_raw, metadata_raw)
    logger.info(f"data:\n{data}")
    logger.info(f"metadata:\n{metadata}")
