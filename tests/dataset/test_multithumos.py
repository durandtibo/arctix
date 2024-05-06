from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import pytest

from arctix.dataset.multithumos import Column, fetch_data, prepare_data, to_array
from arctix.utils.vocab import Vocabulary


class DatasetSplit(NamedTuple):
    name: str
    num_rows: int
    num_examples: int
    seq_len: int


SPLITS = [
    pytest.param(
        DatasetSplit(name="all", num_rows=38_690, num_examples=413, seq_len=1_235), id="all"
    ),
    pytest.param(
        DatasetSplit(name="validation", num_rows=18_482, num_examples=200, seq_len=622),
        id="validation",
    ),
    pytest.param(
        DatasetSplit(name="test", num_rows=20_208, num_examples=213, seq_len=1_235), id="test"
    ),
]


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ.get("ARCTIX_DATA_PATH", tmp_path_factory.mktemp("data"))).joinpath(
        "multithumos"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_multithumos_dataset(data_path: Path, split: DatasetSplit) -> None:
    data_raw = fetch_data(data_path)
    check_data_raw(data_raw)

    data, metadata = prepare_data(data_raw, split=split.name)
    check_data(data, split)
    check_metadata(metadata)

    arrays = to_array(data)
    check_arrays(arrays, split)


def check_data_raw(data: pl.DataFrame) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (38_690, 4)
    assert data.columns == [
        Column.ACTION,
        Column.END_TIME,
        Column.START_TIME,
        Column.VIDEO,
    ]
    assert data.dtypes == [pl.String, pl.Float64, pl.Float64, pl.String]


def check_data(data: pl.DataFrame, split: DatasetSplit) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (split.num_rows, 6)
    assert data.columns == [
        Column.ACTION,
        Column.ACTION_ID,
        Column.END_TIME,
        Column.SPLIT,
        Column.START_TIME,
        Column.VIDEO,
    ]
    assert data.dtypes == [
        pl.String,
        pl.Int64,
        pl.Float32,
        pl.String,
        pl.Float32,
        pl.String,
    ]


def check_metadata(metadata: dict) -> None:
    assert isinstance(metadata, dict)
    assert len(metadata) == 1
    assert isinstance(metadata["vocab_action"], Vocabulary)
    assert len(metadata["vocab_action"]) == 65


def check_arrays(arrays: dict, split: DatasetSplit) -> None:
    assert isinstance(arrays, dict)
    assert len(arrays) == 6

    assert isinstance(arrays[Column.ACTION], np.ma.MaskedArray)
    assert arrays[Column.ACTION].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.ACTION].dtype.kind == "U"

    assert isinstance(arrays[Column.ACTION_ID], np.ma.MaskedArray)
    assert arrays[Column.ACTION_ID].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.ACTION_ID].dtype == np.int64

    assert isinstance(arrays[Column.SEQUENCE_LENGTH], np.ndarray)
    assert arrays[Column.SEQUENCE_LENGTH].shape == (split.num_examples,)
    assert arrays[Column.SEQUENCE_LENGTH].dtype == np.int64

    assert isinstance(arrays[Column.SPLIT], np.ndarray)
    assert arrays[Column.SPLIT].shape == (split.num_examples,)
    assert arrays[Column.SPLIT].dtype.kind == "U"

    assert isinstance(arrays[Column.START_TIME], np.ma.MaskedArray)
    assert arrays[Column.START_TIME].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.START_TIME].dtype == np.float64

    assert isinstance(arrays[Column.END_TIME], np.ma.MaskedArray)
    assert arrays[Column.END_TIME].shape == (split.num_examples, split.seq_len)
    assert arrays[Column.END_TIME].dtype == np.float64
