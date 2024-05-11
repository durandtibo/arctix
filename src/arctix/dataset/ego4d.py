r"""Contain code to prepare the Ego4D data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/ego4d/`.
"""

from __future__ import annotations

__all__ = [
    "Column",
    "load_annotation_file",
    "load_noun_vocab",
    "load_taxonomy_vocab",
    "load_verb_vocab",
]

import logging
from collections import Counter
from pathlib import Path

import polars as pl
from iden.io import load_json

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


def load_annotation_file(path: Path, split: str) -> pl.DataFrame:
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
    >>> from arctix.dataset.ego4d import load_annotation_file
    >>> data = load_annotation_file(
    ...     Path("/path/to/data/ego4d/"), split="train"
    ... )  # doctest: +SKIP

    ```
    """
    path = path.joinpath(f"ego4d_data/v2/annotations/fho_lta_{split}.json")
    logger.info(f"loading data for {split} split...")
    data = load_json(path)
    return (
        pl.DataFrame(
            data["clips"],
            schema={
                Column.ACTION_END_FRAME: pl.Int64,
                Column.ACTION_END_SEC: pl.Float64,
                Column.ACTION_INDEX: pl.Int64,
                Column.ACTION_START_FRAME: pl.Int64,
                Column.ACTION_START_SEC: pl.Float64,
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
                Column.ACTION_INDEX,
                Column.ACTION_START_FRAME,
                Column.ACTION_START_SEC,
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


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("ego4d")
    raw_data = load_annotation_file(path, split="train")
    logger.info(f"data_raw:\n{raw_data}")
