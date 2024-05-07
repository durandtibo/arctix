from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import pytest

from arctix.dataset.epic_kitchen_100 import Column, fetch_data, prepare_data, to_array
from arctix.utils.vocab import Vocabulary


class DatasetSplit(NamedTuple):
    name: str
    num_rows: int
    num_examples: int
    seq_len: int


SPLITS = [
    pytest.param(
        DatasetSplit(name="train", num_rows=67_217, num_examples=495, seq_len=940), id="train"
    ),
    pytest.param(
        DatasetSplit(name="validation", num_rows=9_668, num_examples=138, seq_len=564),
        id="validation",
    ),
]


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ.get("ARCTIX_DATA_PATH", tmp_path_factory.mktemp("data"))).joinpath(
        "epic_kitchen_100"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_epic_kitchen_100_dataset(data_path: Path, split: DatasetSplit) -> None:
    data_raw, metadata_raw = fetch_data(data_path, split=split.name)
    check_data_raw(data_raw, split=split)
    check_metadata(metadata_raw)

    data, metadata = prepare_data(data_raw, metadata_raw)
    check_data(data, split)
    check_metadata(metadata)

    arrays = to_array(data)
    check_arrays(arrays, split)


def check_data_raw(data: pl.DataFrame, split: DatasetSplit) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (split.num_rows, 15)
    assert data.columns == [
        Column.ALL_NOUN_IDS,
        Column.ALL_NOUNS,
        Column.NARRATION,
        Column.NARRATION_ID,
        Column.NARRATION_TIMESTAMP,
        Column.NOUN,
        Column.NOUN_ID,
        Column.PARTICIPANT_ID,
        Column.START_FRAME,
        Column.START_TIMESTAMP,
        Column.STOP_FRAME,
        Column.STOP_TIMESTAMP,
        Column.VERB,
        Column.VERB_ID,
        Column.VIDEO_ID,
    ]
    assert data.dtypes == [
        pl.List(pl.Int64),
        pl.List(pl.String),
        pl.String,
        pl.String,
        pl.Time,
        pl.String,
        pl.Int64,
        pl.String,
        pl.Int64,
        pl.Time,
        pl.Int64,
        pl.Time,
        pl.String,
        pl.Int64,
        pl.String,
    ]


def check_data(data: pl.DataFrame, split: DatasetSplit) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (split.num_rows, 17)
    assert data.columns == [
        Column.ALL_NOUN_IDS,
        Column.ALL_NOUNS,
        Column.NARRATION,
        Column.NARRATION_ID,
        Column.NARRATION_TIMESTAMP,
        Column.NOUN,
        Column.NOUN_ID,
        Column.PARTICIPANT_ID,
        Column.START_FRAME,
        Column.START_TIME_SECOND,
        Column.START_TIMESTAMP,
        Column.STOP_FRAME,
        Column.STOP_TIME_SECOND,
        Column.STOP_TIMESTAMP,
        Column.VERB,
        Column.VERB_ID,
        Column.VIDEO_ID,
    ]
    assert data.dtypes == [
        pl.List(pl.Int64),
        pl.List(pl.String),
        pl.String,
        pl.String,
        pl.Time,
        pl.String,
        pl.Int64,
        pl.String,
        pl.Int64,
        pl.Float32,
        pl.Time,
        pl.Int64,
        pl.Float32,
        pl.Time,
        pl.String,
        pl.Int64,
        pl.String,
    ]


def check_metadata(metadata: dict) -> None:
    assert isinstance(metadata, dict)
    assert len(metadata) == 2

    assert isinstance(metadata["noun_vocab"], Vocabulary)
    assert len(metadata["noun_vocab"]) == 300

    assert isinstance(metadata["verb_vocab"], Vocabulary)
    assert len(metadata["verb_vocab"]) == 97


def check_arrays(arrays: dict, split: DatasetSplit) -> None:
    assert isinstance(arrays, dict)
    assert len(arrays) == 13

    assert isinstance(arrays[Column.NARRATION], np.ma.MaskedArray)
    assert arrays[Column.NARRATION].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.NARRATION].dtype.kind == "U"

    assert isinstance(arrays[Column.NARRATION_ID], np.ma.MaskedArray)
    assert arrays[Column.NARRATION_ID].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.NARRATION_ID].dtype.kind == "U"

    assert isinstance(arrays[Column.NOUN], np.ma.MaskedArray)
    assert arrays[Column.NOUN].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.NOUN].dtype.kind == "U"

    assert isinstance(arrays[Column.NOUN_ID], np.ma.MaskedArray)
    assert arrays[Column.NOUN_ID].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.NOUN_ID].dtype == np.int64

    assert isinstance(arrays[Column.PARTICIPANT_ID], np.ndarray)
    assert arrays[Column.PARTICIPANT_ID].shape == (split.num_examples,)
    assert arrays[Column.PARTICIPANT_ID].dtype.kind == "U"

    assert isinstance(arrays[Column.SEQUENCE_LENGTH], np.ndarray)
    assert arrays[Column.SEQUENCE_LENGTH].shape == (split.num_examples,)
    assert arrays[Column.SEQUENCE_LENGTH].dtype == np.int64

    assert isinstance(arrays[Column.START_FRAME], np.ma.MaskedArray)
    assert arrays[Column.START_FRAME].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.START_FRAME].dtype == np.int64

    assert isinstance(arrays[Column.START_TIME_SECOND], np.ma.MaskedArray)
    assert arrays[Column.START_TIME_SECOND].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.START_TIME_SECOND].dtype == np.float64

    assert isinstance(arrays[Column.STOP_FRAME], np.ma.MaskedArray)
    assert arrays[Column.STOP_FRAME].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.STOP_FRAME].dtype == np.int64

    assert isinstance(arrays[Column.STOP_TIME_SECOND], np.ma.MaskedArray)
    assert arrays[Column.STOP_TIME_SECOND].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.STOP_TIME_SECOND].dtype == np.float64

    assert isinstance(arrays[Column.VERB], np.ma.MaskedArray)
    assert arrays[Column.VERB].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.VERB].dtype.kind == "U"

    assert isinstance(arrays[Column.VERB_ID], np.ma.MaskedArray)
    assert arrays[Column.VERB_ID].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.VERB_ID].dtype == np.int64

    assert isinstance(arrays[Column.VIDEO_ID], np.ndarray)
    assert arrays[Column.VIDEO_ID].shape == (split.num_examples,)
    assert arrays[Column.VIDEO_ID].dtype.kind == "U"
