from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from arctix.dataset.breakfast import Column, fetch_data, prepare_data, to_array
from arctix.utils.vocab import Vocabulary


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ.get("ARCTIX_DATA_PATH", tmp_path_factory.mktemp("data"))).joinpath(
        "breakfast"
    )


def test_breakfast_dataset_coarse(data_path: Path) -> None:
    data_raw = fetch_data(data_path, "segmentation_coarse")
    check_data_raw_coarse(data_raw)

    data, metadata = prepare_data(data_raw)
    check_data_coarse(data)

    arrays = to_array(data)
    check_arrays_coarse(arrays)


def check_data_raw_coarse(data: pl.DataFrame) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (3585, 5)
    assert data.columns == [
        Column.ACTION,
        Column.COOKING_ACTIVITY,
        Column.END_TIME,
        Column.PERSON,
        Column.START_TIME,
    ]
    assert data.dtypes == [pl.String, pl.String, pl.Float64, pl.String, pl.Float64]


def check_data_coarse(data: pl.DataFrame) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (3585, 8)
    assert data.columns == [
        Column.ACTION,
        Column.ACTION_ID,
        Column.COOKING_ACTIVITY,
        Column.COOKING_ACTIVITY_ID,
        Column.END_TIME,
        Column.PERSON,
        Column.PERSON_ID,
        Column.START_TIME,
    ]
    assert data.dtypes == [
        pl.String,
        pl.Int64,
        pl.String,
        pl.Int64,
        pl.Float32,
        pl.String,
        pl.Int64,
        pl.Float32,
    ]


def check_metadata_coarse(metadata: dict) -> None:
    assert isinstance(metadata, dict)
    assert len(metadata) == 3
    assert isinstance(metadata["vocab_action"], Vocabulary)
    assert len(metadata["vocab_action"]) == 48
    assert isinstance(metadata["vocab_activity"], Vocabulary)
    assert len(metadata["vocab_activity"]) == 10
    assert isinstance(metadata["vocab_person"], Vocabulary)
    assert len(metadata["vocab_person"]) == 52


def check_arrays_coarse(arrays: dict) -> None:
    assert isinstance(arrays, dict)
    assert len(arrays) == 9

    assert isinstance(arrays[Column.ACTION], np.ma.MaskedArray)
    assert arrays[Column.ACTION].shape == (503, 25)
    assert arrays[Column.ACTION].dtype.kind == "U"

    assert isinstance(arrays[Column.ACTION_ID], np.ma.MaskedArray)
    assert arrays[Column.ACTION_ID].shape == (503, 25)
    assert arrays[Column.ACTION_ID].dtype == np.int64

    assert isinstance(arrays[Column.COOKING_ACTIVITY], np.ndarray)
    assert arrays[Column.COOKING_ACTIVITY].shape == (503,)
    assert arrays[Column.COOKING_ACTIVITY].dtype.kind == "U"

    assert isinstance(arrays[Column.COOKING_ACTIVITY_ID], np.ndarray)
    assert arrays[Column.COOKING_ACTIVITY_ID].shape == (503,)
    assert arrays[Column.COOKING_ACTIVITY_ID].dtype == np.int64

    assert isinstance(arrays[Column.PERSON], np.ndarray)
    assert arrays[Column.PERSON].shape == (503,)
    assert arrays[Column.PERSON].dtype.kind == "U"

    assert isinstance(arrays[Column.PERSON_ID], np.ndarray)
    assert arrays[Column.PERSON_ID].shape == (503,)
    assert arrays[Column.PERSON_ID].dtype == np.int64

    assert isinstance(arrays[Column.SEQUENCE_LENGTH], np.ndarray)
    assert arrays[Column.SEQUENCE_LENGTH].shape == (503,)
    assert arrays[Column.SEQUENCE_LENGTH].dtype == np.int64

    assert isinstance(arrays[Column.START_TIME], np.ma.MaskedArray)
    assert arrays[Column.START_TIME].shape == (503, 25)
    assert arrays[Column.START_TIME].dtype == np.float64

    assert isinstance(arrays[Column.END_TIME], np.ma.MaskedArray)
    assert arrays[Column.END_TIME].shape == (503, 25)
    assert arrays[Column.END_TIME].dtype == np.float64


def test_breakfast_dataset_fine(data_path: Path) -> None:
    data_raw = fetch_data(data_path, "segmentation_fine")
    check_data_raw_fine(data_raw)

    data, metadata = prepare_data(data_raw)
    check_data_fine(data)

    arrays = to_array(data)
    check_arrays_fine(arrays)


def check_data_raw_fine(data: pl.DataFrame) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (10_715, 5)
    assert data.columns == [
        Column.ACTION,
        Column.COOKING_ACTIVITY,
        Column.END_TIME,
        Column.PERSON,
        Column.START_TIME,
    ]
    assert data.dtypes == [pl.String, pl.String, pl.Float64, pl.String, pl.Float64]


def check_data_fine(data: pl.DataFrame) -> None:
    assert isinstance(data, pl.DataFrame)
    assert data.shape == (10_715, 8)
    assert data.columns == [
        Column.ACTION,
        Column.ACTION_ID,
        Column.COOKING_ACTIVITY,
        Column.COOKING_ACTIVITY_ID,
        Column.END_TIME,
        Column.PERSON,
        Column.PERSON_ID,
        Column.START_TIME,
    ]
    assert data.dtypes == [
        pl.String,
        pl.Int64,
        pl.String,
        pl.Int64,
        pl.Float32,
        pl.String,
        pl.Int64,
        pl.Float32,
    ]


def check_metadata_fine(metadata: dict) -> None:
    assert isinstance(metadata, dict)
    assert len(metadata) == 3
    assert isinstance(metadata["vocab_action"], Vocabulary)
    assert len(metadata["vocab_action"]) == 178
    assert isinstance(metadata["vocab_activity"], Vocabulary)
    assert len(metadata["vocab_activity"]) == 10
    assert isinstance(metadata["vocab_person"], Vocabulary)
    assert len(metadata["vocab_person"]) == 49


def check_arrays_fine(arrays: dict) -> None:
    assert isinstance(arrays, dict)
    assert len(arrays) == 9

    assert isinstance(arrays[Column.ACTION], np.ma.MaskedArray)
    assert arrays[Column.ACTION].shape == (257, 165)
    assert arrays[Column.ACTION].dtype.kind == "U"

    assert isinstance(arrays[Column.ACTION_ID], np.ma.MaskedArray)
    assert arrays[Column.ACTION_ID].shape == (257, 165)
    assert arrays[Column.ACTION_ID].dtype == np.int64

    assert isinstance(arrays[Column.COOKING_ACTIVITY], np.ndarray)
    assert arrays[Column.COOKING_ACTIVITY].shape == (257,)
    assert arrays[Column.COOKING_ACTIVITY].dtype.kind == "U"

    assert isinstance(arrays[Column.COOKING_ACTIVITY_ID], np.ndarray)
    assert arrays[Column.COOKING_ACTIVITY_ID].shape == (257,)
    assert arrays[Column.COOKING_ACTIVITY_ID].dtype == np.int64

    assert isinstance(arrays[Column.PERSON], np.ndarray)
    assert arrays[Column.PERSON].shape == (257,)
    assert arrays[Column.PERSON].dtype.kind == "U"

    assert isinstance(arrays[Column.PERSON_ID], np.ndarray)
    assert arrays[Column.PERSON_ID].shape == (257,)
    assert arrays[Column.PERSON_ID].dtype == np.int64

    assert isinstance(arrays[Column.SEQUENCE_LENGTH], np.ndarray)
    assert arrays[Column.SEQUENCE_LENGTH].shape == (257,)
    assert arrays[Column.SEQUENCE_LENGTH].dtype == np.int64

    assert isinstance(arrays[Column.START_TIME], np.ma.MaskedArray)
    assert arrays[Column.START_TIME].shape == (257, 165)
    assert arrays[Column.START_TIME].dtype == np.float64

    assert isinstance(arrays[Column.END_TIME], np.ma.MaskedArray)
    assert arrays[Column.END_TIME].shape == (257, 165)
    assert arrays[Column.END_TIME].dtype == np.float64
