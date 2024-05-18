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

import numpy as np
import polars as pl
from iden.io import load_json

from arctix.transformer import dataframe as td
from arctix.utils.masking import convert_sequences_to_array, generate_mask_from_lengths
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
    ACTION_START_SEC_DIFF: str = "action_clip_start_sec_diff"
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


def prepare_data(
    frame: pl.DataFrame, metadata: dict, group_col: str = Column.CLIP_ID
) -> tuple[pl.DataFrame, dict]:
    r"""Prepare the data.

    Args:
        frame: The raw DataFrame.
        metadata: The metadata wich contains the vocabularies to
            convert verbs and nouns to index.
        group_col: The column used to generate the sequences.

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
    shape: (5, 13)
    ┌─────┬─────┬─────┬────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ act ┆ act ┆ act ┆ action ┆ actio ┆ actio ┆ clip_ ┆ noun  ┆ noun_ ┆ split ┆ verb  ┆ verb_ ┆ video │
    │ ion ┆ ion ┆ ion ┆ _clip_ ┆ n_cli ┆ n_idx ┆ uid   ┆ ---   ┆ label ┆ ---   ┆ ---   ┆ label ┆ _uid  │
    │ _cl ┆ _cl ┆ _cl ┆ start_ ┆ p_sta ┆ ---   ┆ ---   ┆ str   ┆ ---   ┆ str   ┆ str   ┆ ---   ┆ ---   │
    │ ip_ ┆ ip_ ┆ ip_ ┆ sec    ┆ rt_se ┆ i64   ┆ str   ┆       ┆ i64   ┆       ┆       ┆ i64   ┆ str   │
    │ end ┆ end ┆ sta ┆ ---    ┆ c_dif ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ _fr ┆ _se ┆ rt_ ┆ f64    ┆ f     ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ ame ┆ c   ┆ fra ┆        ┆ ---   ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ --- ┆ --- ┆ me  ┆        ┆ f64   ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ i64 ┆ f64 ┆ --- ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │     ┆     ┆ i64 ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    ╞═════╪═════╪═════╪════════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
    │ 47  ┆ 4.7 ┆ 23  ┆ 2.3    ┆ 0.0   ┆ 0     ┆ clip1 ┆ noun2 ┆ 2     ┆ train ┆ verb4 ┆ 4     ┆ video │
    │     ┆     ┆     ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 82  ┆ 8.2 ┆ 39  ┆ 3.9    ┆ 1.6   ┆ 1     ┆ clip1 ┆ noun3 ┆ 3     ┆ train ┆ verb2 ┆ 2     ┆ video │
    │     ┆     ┆     ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 102 ┆ 10. ┆ 74  ┆ 7.4    ┆ 3.5   ┆ 2     ┆ clip1 ┆ noun1 ┆ 1     ┆ train ┆ verb1 ┆ 1     ┆ video │
    │     ┆ 2   ┆     ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 1     │
    │ 74  ┆ 7.4 ┆ 12  ┆ 1.2    ┆ 0.0   ┆ 0     ┆ clip2 ┆ noun1 ┆ 1     ┆ train ┆ verb1 ┆ 1     ┆ video │
    │     ┆     ┆     ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 2     │
    │ 142 ┆ 14. ┆ 82  ┆ 8.2    ┆ 7.0   ┆ 1     ┆ clip2 ┆ noun2 ┆ 2     ┆ train ┆ verb2 ┆ 2     ┆ video │
    │     ┆ 2   ┆     ┆        ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆       ┆ 2     │
    └─────┴─────┴─────┴────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
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
            td.TimeDiff(
                group_cols=[group_col],
                time_col=Column.ACTION_START_SEC,
                time_diff_col=Column.ACTION_START_SEC_DIFF,
            ),
            td.Sort(columns=[group_col, Column.ACTION_INDEX]),
            td.SortColumns(),
        ]
    )
    out = transformer.transform(frame)
    return out, metadata


def group_by_sequence(frame: pl.DataFrame, group_col: str = Column.CLIP_ID) -> pl.DataFrame:
    r"""Group the DataFrame by sequences of actions.

    Args:
        frame: The input DataFrame.
        group_col: The column used to generate the sequences.

    Returns:
        The DataFrame after the grouping.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.ego4d import Column, group_by_sequence
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
    ...         Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
    ...         Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
    ...         Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
    ...         Column.ACTION_START_SEC_DIFF: [0.0, 1.6, 3.5, 0.0, 7.0],
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
    >>> data = group_by_sequence(frame)
    >>> with pl.Config(tbl_cols=-1):
    ...     data
    shape: (2, 12)
    ┌────────┬────────┬────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ action ┆ action ┆ action ┆ actio ┆ actio ┆ clip_ ┆ noun  ┆ noun_ ┆ seque ┆ split ┆ verb  ┆ verb_ │
    │ _clip_ ┆ _clip_ ┆ _clip_ ┆ n_cli ┆ n_cli ┆ uid   ┆ ---   ┆ label ┆ nce_l ┆ ---   ┆ ---   ┆ label │
    │ end_fr ┆ end_se ┆ start_ ┆ p_sta ┆ p_sta ┆ ---   ┆ list[ ┆ ---   ┆ ength ┆ str   ┆ list[ ┆ ---   │
    │ ame    ┆ c      ┆ frame  ┆ rt_se ┆ rt_se ┆ str   ┆ str]  ┆ list[ ┆ ---   ┆       ┆ str]  ┆ list[ │
    │ ---    ┆ ---    ┆ ---    ┆ c     ┆ c_dif ┆       ┆       ┆ i64]  ┆ i64   ┆       ┆       ┆ i64]  │
    │ list[i ┆ list[f ┆ list[i ┆ ---   ┆ f     ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │ 64]    ┆ 64]    ┆ 64]    ┆ list[ ┆ ---   ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │        ┆        ┆        ┆ f64]  ┆ list[ ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    │        ┆        ┆        ┆       ┆ f64]  ┆       ┆       ┆       ┆       ┆       ┆       ┆       │
    ╞════════╪════════╪════════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
    │ [47,   ┆ [4.7,  ┆ [23,   ┆ [2.3, ┆ [0.0, ┆ clip1 ┆ ["nou ┆ [2,   ┆ 3     ┆ train ┆ ["ver ┆ [4,   │
    │ 82,    ┆ 8.2,   ┆ 39,    ┆ 3.9,  ┆ 1.6,  ┆       ┆ n2",  ┆ 3, 1] ┆       ┆       ┆ b4",  ┆ 2, 1] │
    │ 102]   ┆ 10.2]  ┆ 74]    ┆ 7.4]  ┆ 3.5]  ┆       ┆ "noun ┆       ┆       ┆       ┆ "verb ┆       │
    │        ┆        ┆        ┆       ┆       ┆       ┆ 3",   ┆       ┆       ┆       ┆ 2",   ┆       │
    │        ┆        ┆        ┆       ┆       ┆       ┆ "noun ┆       ┆       ┆       ┆ "verb ┆       │
    │        ┆        ┆        ┆       ┆       ┆       ┆ 1"]   ┆       ┆       ┆       ┆ 1"]   ┆       │
    │ [74,   ┆ [7.4,  ┆ [12,   ┆ [1.2, ┆ [0.0, ┆ clip2 ┆ ["nou ┆ [1,   ┆ 2     ┆ train ┆ ["ver ┆ [1,   │
    │ 142]   ┆ 14.2]  ┆ 82]    ┆ 8.2]  ┆ 7.0]  ┆       ┆ n1",  ┆ 2]    ┆       ┆       ┆ b1",  ┆ 2]    │
    │        ┆        ┆        ┆       ┆       ┆       ┆ "noun ┆       ┆       ┆       ┆ "verb ┆       │
    │        ┆        ┆        ┆       ┆       ┆       ┆ 2"]   ┆       ┆       ┆       ┆ 2"]   ┆       │
    └────────┴────────┴────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

    ```
    """
    data = (
        frame.sort(by=[Column.VIDEO_ID, Column.CLIP_ID, Column.ACTION_INDEX])
        .group_by([group_col])
        .agg(
            pl.col(Column.ACTION_END_FRAME),
            pl.col(Column.ACTION_END_SEC),
            pl.col(Column.ACTION_START_FRAME),
            pl.col(Column.ACTION_START_SEC),
            pl.col(Column.ACTION_START_SEC_DIFF),
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


def to_array(frame: pl.DataFrame, group_col: str = Column.CLIP_ID) -> dict[str, np.ndarray]:
    r"""Convert a DataFrame to a dictionary of arrays.

    Args:
        frame: The input DataFrame.
        group_col: The column used to generate the sequences.

    Returns:
        The dictionary of arrays.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.ego4d import Column, to_array
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
    ...         Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
    ...         Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
    ...         Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
    ...         Column.ACTION_START_SEC_DIFF: [0.0, 1.6, 3.5, 0.0, 7.0],
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
    >>> arrays = to_array(frame)
    >>> arrays
    {'noun': masked_array(
      data=[['noun2', 'noun3', 'noun1'],
            ['noun1', 'noun2', --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value='N/A',
      dtype='<U5'),
      'noun_label': masked_array(
      data=[[2, 3, 1],
            [1, 2, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999),
      'split': array(['train', 'train'], dtype='<U5'),
      'sequence_length': array([3, 2]),
      'action_clip_start_frame': masked_array(
      data=[[23, 39, 74],
            [12, 82, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999),
      'action_clip_start_sec': masked_array(
      data=[[2.3, 3.9, 7.4],
            [1.2, 8.2, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=1e+20),
      'action_clip_start_sec_diff': masked_array(
      data=[[0.0, 1.6, 3.5],
            [0.0, 7.0, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=1e+20),
      'action_clip_end_frame': masked_array(
      data=[[47, 82, 102],
            [74, 142, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999),
      'action_clip_end_sec': masked_array(
      data=[[4.7, 8.2, 10.2],
            [7.4, 14.2, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=1e+20),
      'verb': masked_array(
      data=[['verb4', 'verb2', 'verb1'],
            ['verb1', 'verb2', --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value='N/A',
      dtype='<U5'),
      'verb_label': masked_array(
      data=[[4, 2, 1],
            [1, 2, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999),
      'clip_uid': array(['clip1', 'clip2'], dtype='<U5')}

    ```
    """
    groups = group_by_sequence(frame, group_col)
    lengths = groups.get_column(Column.SEQUENCE_LENGTH).to_numpy()
    mask = generate_mask_from_lengths(lengths)
    return {
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
        Column.SPLIT: groups.get_column(Column.SPLIT).to_numpy().astype(str),
        Column.SEQUENCE_LENGTH: groups.get_column(Column.SEQUENCE_LENGTH)
        .to_numpy()
        .astype(np.int64),
        Column.ACTION_START_FRAME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_START_FRAME).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.ACTION_START_SEC: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_START_SEC).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.ACTION_START_SEC_DIFF: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_START_SEC_DIFF).to_list(),
                max_len=mask.shape[1],
                dtype=np.float64,
                padded_value=-1.0,
            ),
            mask=mask,
        ),
        Column.ACTION_END_FRAME: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_END_FRAME).to_list(),
                max_len=mask.shape[1],
                dtype=np.int64,
                padded_value=-1,
            ),
            mask=mask,
        ),
        Column.ACTION_END_SEC: np.ma.masked_array(
            data=convert_sequences_to_array(
                groups.get_column(Column.ACTION_END_SEC).to_list(),
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
        group_col: groups.get_column(group_col).to_numpy().astype(str),
    }


def to_list(frame: pl.DataFrame, group_col: str = Column.CLIP_ID) -> dict[str, list]:
    r"""Convert a DataFrame to a dictionary of lists.

    Args:
        frame: The input DataFrame.
        group_col: The column used to generate the sequences.

    Returns:
        The dictionary of lists.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arctix.dataset.ego4d import Column, to_list
    >>> frame = pl.DataFrame(
    ...     {
    ...         Column.ACTION_END_FRAME: [47, 82, 102, 74, 142],
    ...         Column.ACTION_END_SEC: [4.7, 8.2, 10.2, 7.4, 14.2],
    ...         Column.ACTION_START_FRAME: [23, 39, 74, 12, 82],
    ...         Column.ACTION_START_SEC: [2.3, 3.9, 7.4, 1.2, 8.2],
    ...         Column.ACTION_START_SEC_DIFF: [0.0, 1.6, 3.5, 0.0, 7.0],
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
    >>> data_list = to_list(frame)
    >>> data_list
    {'action_clip_end_frame': [[47, 82, 102], [74, 142]],
     'action_clip_end_sec': [[4.7, 8.2, 10.2], [7.4, 14.2]],
     'action_clip_start_frame': [[23, 39, 74], [12, 82]],
     'action_clip_start_sec': [[2.3, 3.9, 7.4], [1.2, 8.2]],
     'action_clip_start_sec_diff': [[0.0, 1.6, 3.5], [0.0, 7.0]],
     'clip_uid': ['clip1', 'clip2'],
     'noun': [['noun2', 'noun3', 'noun1'], ['noun1', 'noun2']],
     'noun_label': [[2, 3, 1], [1, 2]],
     'sequence_length': [3, 2],
     'split': ['train', 'train'],
     'verb': [['verb4', 'verb2', 'verb1'], ['verb1', 'verb2']],
     'verb_label': [[4, 2, 1], [1, 2]]}

    ```
    """
    return group_by_sequence(frame, group_col).to_dict(as_series=False)


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

    arrays = to_array(data)
    logger.info(f"arrays:\n{arrays}")
