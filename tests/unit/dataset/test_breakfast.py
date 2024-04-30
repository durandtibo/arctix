from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from iden.io import save_text
from polars.testing import assert_frame_equal

from arctix.dataset.breakfast import (
    URLS,
    Column,
    download_data,
    fetch_data,
    filter_by_split,
    group_by_sequence,
    load_annotation_file,
    load_data,
    parse_annotation_lines,
    prepare_data,
    to_array,
    to_list,
)
from arctix.utils.vocab import Vocabulary


@pytest.fixture(scope="module")
def data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("P03_cam01_P03_cereals.txt")
    save_text(
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path,
    )
    return path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("dataset")
    save_text(
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("segmentation_coarse/P03_cam01_P03_cereals.txt"),
    )
    save_text(  # duplicate example
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("segmentation_coarse/P03_cam02_P03_cereals.txt"),
    )
    save_text(
        "1-47 SIL  \n48-215 pour_milk  \n216-565 spoon_powder  \n566-747 SIL  \n",
        path.joinpath("segmentation_coarse/milk/P54_webcam02_P54_milk.txt"),
    )
    return path


@pytest.fixture()
def data_raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION: [
                "SIL",
                "take_bowl",
                "pour_cereals",
                "pour_milk",
                "stir_cereals",
                "SIL",
                "SIL",
                "pour_milk",
                "spoon_powder",
                "SIL",
            ],
            Column.COOKING_ACTIVITY: [
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "milk",
                "milk",
                "milk",
                "milk",
            ],
            Column.END_TIME: [
                30.0,
                150.0,
                428.0,
                575.0,
                705.0,
                836.0,
                47.0,
                215.0,
                565.0,
                747.0,
            ],
            Column.PERSON: [
                "P03",
                "P03",
                "P03",
                "P03",
                "P03",
                "P03",
                "P54",
                "P54",
                "P54",
                "P54",
            ],
            Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
        },
        schema={
            Column.ACTION: pl.String,
            Column.COOKING_ACTIVITY: pl.String,
            Column.END_TIME: pl.Float64,
            Column.PERSON: pl.String,
            Column.START_TIME: pl.Float64,
        },
    )


@pytest.fixture()
def data_prepared() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION: [
                "SIL",
                "take_bowl",
                "pour_cereals",
                "pour_milk",
                "stir_cereals",
                "SIL",
                "SIL",
                "pour_milk",
                "spoon_powder",
                "SIL",
            ],
            Column.ACTION_ID: [0, 2, 5, 1, 3, 0, 0, 1, 4, 0],
            Column.COOKING_ACTIVITY: [
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "milk",
                "milk",
                "milk",
                "milk",
            ],
            Column.COOKING_ACTIVITY_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            Column.END_TIME: [
                30.0,
                150.0,
                428.0,
                575.0,
                705.0,
                836.0,
                47.0,
                215.0,
                565.0,
                747.0,
            ],
            Column.PERSON: ["P03", "P03", "P03", "P03", "P03", "P03", "P54", "P54", "P54", "P54"],
            Column.PERSON_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
        },
        schema={
            Column.ACTION: pl.String,
            Column.ACTION_ID: pl.Int64,
            Column.COOKING_ACTIVITY: pl.String,
            Column.COOKING_ACTIVITY_ID: pl.Int64,
            Column.END_TIME: pl.Float32,
            Column.PERSON: pl.String,
            Column.PERSON_ID: pl.Int64,
            Column.START_TIME: pl.Float32,
        },
    )


@pytest.fixture()
def data_prepared_empty() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION: [],
            Column.ACTION_ID: [],
            Column.COOKING_ACTIVITY: [],
            Column.COOKING_ACTIVITY_ID: [],
            Column.END_TIME: [],
            Column.PERSON: [],
            Column.PERSON_ID: [],
            Column.START_TIME: [],
        },
        schema={
            Column.ACTION: pl.String,
            Column.ACTION_ID: pl.Int64,
            Column.COOKING_ACTIVITY: pl.String,
            Column.COOKING_ACTIVITY_ID: pl.Int64,
            Column.END_TIME: pl.Float32,
            Column.PERSON: pl.String,
            Column.PERSON_ID: pl.Int64,
            Column.START_TIME: pl.Float32,
        },
    )


################################
#     Tests for fetch_data     #
################################


def test_fetch_data_remove_duplicate_examples(data_dir: Path, data_raw: pl.DataFrame) -> None:
    with patch("arctix.dataset.breakfast.download_data") as download_mock:
        data = fetch_data(data_dir, name="segmentation_coarse")
        download_mock.assert_called_once_with(data_dir, False)
    assert_frame_equal(data, data_raw)


def test_fetch_data_keep_duplicate_examples(data_dir: Path) -> None:
    with patch("arctix.dataset.breakfast.download_data") as download_mock:
        data = fetch_data(
            data_dir, name="segmentation_coarse", remove_duplicate=False, force_download=True
        )
        download_mock.assert_called_once_with(data_dir, True)
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION: [
                    "SIL",
                    "SIL",
                    "take_bowl",
                    "take_bowl",
                    "pour_cereals",
                    "pour_cereals",
                    "pour_milk",
                    "pour_milk",
                    "stir_cereals",
                    "stir_cereals",
                    "SIL",
                    "SIL",
                    "SIL",
                    "pour_milk",
                    "spoon_powder",
                    "SIL",
                ],
                Column.START_TIME: [
                    1.0,
                    1.0,
                    31.0,
                    31.0,
                    151.0,
                    151.0,
                    429.0,
                    429.0,
                    576.0,
                    576.0,
                    706.0,
                    706.0,
                    1.0,
                    48.0,
                    216.0,
                    566.0,
                ],
                Column.END_TIME: [
                    30.0,
                    30.0,
                    150.0,
                    150.0,
                    428.0,
                    428.0,
                    575.0,
                    575.0,
                    705.0,
                    705.0,
                    836.0,
                    836.0,
                    47.0,
                    215.0,
                    565.0,
                    747.0,
                ],
                Column.COOKING_ACTIVITY: [
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "milk",
                    "milk",
                    "milk",
                    "milk",
                ],
                Column.PERSON: [
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P54",
                    "P54",
                    "P54",
                    "P54",
                ],
            }
        ),
        check_column_order=False,
    )


def test_fetch_data_incorrect_name(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        fetch_data(tmp_path, "incorrect")


###################################
#     Tests for download_data     #
###################################


def test_download_data(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_data(tmp_path)
        assert download_mock.call_args_list == [
            call(
                URLS["segmentation_coarse"],
                tmp_path.joinpath("segmentation_coarse.tar.gz"),
                quiet=False,
                fuzzy=True,
            ),
            call(
                URLS["segmentation_fine"],
                tmp_path.joinpath("segmentation_fine.tar.gz"),
                quiet=False,
                fuzzy=True,
            ),
        ]
        assert tarfile_mock.call_args_list == [
            call(tmp_path.joinpath("segmentation_coarse.tar.gz")),
            call(tmp_path.joinpath("segmentation_fine.tar.gz")),
        ]


def test_download_data_dir_exists(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.Path.is_dir", lambda *args, **kwargs: True),  # noqa: ARG005
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_data(tmp_path)
        download_mock.assert_not_called()
        tarfile_mock.assert_not_called()


def test_download_data_dir_exists_force_download(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.Path.is_dir", lambda *args, **kwargs: True),  # noqa: ARG005
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_data(tmp_path, force_download=True)
        assert download_mock.call_args_list == [
            call(
                URLS["segmentation_coarse"],
                tmp_path.joinpath("segmentation_coarse.tar.gz"),
                quiet=False,
                fuzzy=True,
            ),
            call(
                URLS["segmentation_fine"],
                tmp_path.joinpath("segmentation_fine.tar.gz"),
                quiet=False,
                fuzzy=True,
            ),
        ]
        assert tarfile_mock.call_args_list == [
            call(tmp_path.joinpath("segmentation_coarse.tar.gz")),
            call(tmp_path.joinpath("segmentation_fine.tar.gz")),
        ]


###############################
#     Tests for load_data     #
###############################


def test_load_data(data_dir: Path, data_raw: pl.DataFrame) -> None:
    assert_frame_equal(load_data(data_dir), data_raw)


def test_load_data_keep_duplicates(data_dir: Path) -> None:
    assert_frame_equal(
        load_data(data_dir, remove_duplicate=False),
        pl.DataFrame(
            {
                Column.ACTION: [
                    "SIL",
                    "SIL",
                    "take_bowl",
                    "take_bowl",
                    "pour_cereals",
                    "pour_cereals",
                    "pour_milk",
                    "pour_milk",
                    "stir_cereals",
                    "stir_cereals",
                    "SIL",
                    "SIL",
                    "SIL",
                    "pour_milk",
                    "spoon_powder",
                    "SIL",
                ],
                Column.COOKING_ACTIVITY: [
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "milk",
                    "milk",
                    "milk",
                    "milk",
                ],
                Column.END_TIME: [
                    30.0,
                    30.0,
                    150.0,
                    150.0,
                    428.0,
                    428.0,
                    575.0,
                    575.0,
                    705.0,
                    705.0,
                    836.0,
                    836.0,
                    47.0,
                    215.0,
                    565.0,
                    747.0,
                ],
                Column.PERSON: [
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P54",
                    "P54",
                    "P54",
                    "P54",
                ],
                Column.START_TIME: [
                    1.0,
                    1.0,
                    31.0,
                    31.0,
                    151.0,
                    151.0,
                    429.0,
                    429.0,
                    576.0,
                    576.0,
                    706.0,
                    706.0,
                    1.0,
                    48.0,
                    216.0,
                    566.0,
                ],
            },
            schema={
                Column.ACTION: pl.String,
                Column.COOKING_ACTIVITY: pl.String,
                Column.END_TIME: pl.Float64,
                Column.PERSON: pl.String,
                Column.START_TIME: pl.Float64,
            },
        ),
    )


##########################################
#     Tests for load_annotation_file     #
##########################################


def test_load_annotation_file_incorrect_extension() -> None:
    with pytest.raises(ValueError, match="Incorrect file extension."):
        load_annotation_file(Mock(spec=Path))


def test_load_annotation_file(data_file: Path) -> None:
    assert objects_are_equal(
        load_annotation_file(data_file),
        {
            Column.ACTION: ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"],
            Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0],
            Column.END_TIME: [30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
            Column.COOKING_ACTIVITY: [
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
                "cereals",
            ],
            Column.PERSON: ["P03", "P03", "P03", "P03", "P03", "P03"],
        },
    )


############################################
#     Tests for parse_annotation_lines     #
############################################


def test_parse_annotation_lines_empty() -> None:
    assert objects_are_equal(
        parse_annotation_lines([]),
        {
            Column.ACTION: [],
            Column.START_TIME: [],
            Column.END_TIME: [],
        },
    )


def test_parse_annotation_lines() -> None:
    assert objects_are_equal(
        parse_annotation_lines(
            [
                "1-30 SIL  \n",
                "31-150 take_bowl  \n",
                "151-428 pour_cereals  \n",
                "429-575 pour_milk",
                "576-705 stir_cereals  \n",
                "706-836 SIL  \n",
            ]
        ),
        {
            Column.ACTION: ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"],
            Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0],
            Column.END_TIME: [30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
        },
    )


#####################################
#     Tests for filter_by_split     #
#####################################


def test_filter_by_split_train1() -> None:
    out = filter_by_split(
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P29", "P41", "P42", "P54"],
                "col": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ),
        split="train1",
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                Column.PERSON: ["P16", "P28", "P29", "P41", "P42", "P54"],
                "col": [3, 4, 5, 6, 7, 8],
            }
        ),
    )


def test_filter_by_split_train2() -> None:
    out = filter_by_split(
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P29", "P41", "P42", "P54"],
                "col": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ),
        split="train2",
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P29", "P41", "P42", "P54"],
                "col": [1, 2, 5, 6, 7, 8],
            }
        ),
    )


def test_filter_by_split_train3() -> None:
    out = filter_by_split(
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P29", "P41", "P42", "P54"],
                "col": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ),
        split="train3",
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P42", "P54"],
                "col": [1, 2, 3, 4, 7, 8],
            }
        ),
    )


def test_filter_by_split_train4() -> None:
    out = filter_by_split(
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P29", "P41", "P42", "P54"],
                "col": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ),
        split="train4",
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                Column.PERSON: ["P03", "P15", "P16", "P28", "P29", "P41"],
                "col": [1, 2, 3, 4, 5, 6],
            }
        ),
    )


def test_filter_by_split_empty() -> None:
    out = filter_by_split(pl.DataFrame({Column.PERSON: []}))
    assert_frame_equal(out, pl.DataFrame({Column.PERSON: []}))


##################################
#     Tests for prepare_data     #
##################################


def test_prepare_data(data_raw: pl.DataFrame, data_prepared: pl.DataFrame) -> None:
    data, metadata = prepare_data(data_raw)
    assert_frame_equal(data, data_prepared)
    assert objects_are_equal(
        metadata,
        {
            "vocab_action": Vocabulary(
                Counter(
                    {
                        "SIL": 4,
                        "pour_milk": 2,
                        "take_bowl": 1,
                        "stir_cereals": 1,
                        "spoon_powder": 1,
                        "pour_cereals": 1,
                    }
                )
            ),
            "vocab_activity": Vocabulary(Counter({"cereals": 6, "milk": 4})),
            "vocab_person": Vocabulary(Counter({"P03": 6, "P54": 4})),
        },
    )


def test_prepare_data_split_train1() -> None:
    data, metadata = prepare_data(
        pl.DataFrame(
            {
                Column.ACTION: [
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                    "SIL",
                    "pour_milk",
                    "spoon_powder",
                    "SIL",
                ],
                Column.START_TIME: [1.0, 31.0, 151.0, 429.0, 576.0, 706.0, 1.0, 48.0, 216.0, 566.0],
                Column.END_TIME: [
                    30.0,
                    150.0,
                    428.0,
                    575.0,
                    705.0,
                    836.0,
                    47.0,
                    215.0,
                    565.0,
                    747.0,
                ],
                Column.COOKING_ACTIVITY: [
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "cereals",
                    "milk",
                    "milk",
                    "milk",
                    "milk",
                ],
                Column.PERSON: [
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P03",
                    "P54",
                    "P54",
                    "P54",
                    "P54",
                ],
            },
            schema={
                Column.ACTION: pl.String,
                Column.COOKING_ACTIVITY: pl.String,
                Column.END_TIME: pl.Float64,
                Column.PERSON: pl.String,
                Column.START_TIME: pl.Float64,
            },
        ),
        split="train1",
    )
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION: ["SIL", "pour_milk", "spoon_powder", "SIL"],
                Column.ACTION_ID: [0, 1, 4, 0],
                Column.COOKING_ACTIVITY: ["milk", "milk", "milk", "milk"],
                Column.COOKING_ACTIVITY_ID: [1, 1, 1, 1],
                Column.END_TIME: [47.0, 215.0, 565.0, 747.0],
                Column.PERSON: ["P54", "P54", "P54", "P54"],
                Column.PERSON_ID: [1, 1, 1, 1],
                Column.START_TIME: [1.0, 48.0, 216.0, 566.0],
            },
            schema={
                Column.ACTION: pl.String,
                Column.ACTION_ID: pl.Int64,
                Column.COOKING_ACTIVITY: pl.String,
                Column.COOKING_ACTIVITY_ID: pl.Int64,
                Column.END_TIME: pl.Float32,
                Column.PERSON: pl.String,
                Column.PERSON_ID: pl.Int64,
                Column.START_TIME: pl.Float32,
            },
        ),
    )
    assert objects_are_equal(
        metadata,
        {
            "vocab_action": Vocabulary(
                Counter(
                    {
                        "SIL": 4,
                        "pour_milk": 2,
                        "take_bowl": 1,
                        "stir_cereals": 1,
                        "spoon_powder": 1,
                        "pour_cereals": 1,
                    }
                )
            ),
            "vocab_activity": Vocabulary(Counter({"cereals": 6, "milk": 4})),
            "vocab_person": Vocabulary(Counter({"P03": 6, "P54": 4})),
        },
    )


def test_prepare_data_empty(data_prepared_empty: pl.DataFrame) -> None:
    data, metadata = prepare_data(
        pl.DataFrame(
            {
                Column.ACTION: [],
                Column.COOKING_ACTIVITY: [],
                Column.END_TIME: [],
                Column.PERSON: [],
                Column.START_TIME: [],
            },
            schema={
                Column.ACTION: pl.String,
                Column.COOKING_ACTIVITY: pl.String,
                Column.END_TIME: pl.Float32,
                Column.PERSON: pl.String,
                Column.START_TIME: pl.Float32,
            },
        )
    )
    assert_frame_equal(data, data_prepared_empty)
    assert objects_are_equal(
        metadata,
        {
            "vocab_action": Vocabulary(Counter({})),
            "vocab_activity": Vocabulary(Counter({})),
            "vocab_person": Vocabulary(Counter({})),
        },
    )


#######################################
#     Tests for group_by_sequence     #
#######################################


def test_group_by_sequence(data_prepared: pl.DataFrame) -> None:
    assert_frame_equal(
        group_by_sequence(data_prepared),
        pl.DataFrame(
            {
                Column.ACTION: [
                    ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"],
                    ["SIL", "pour_milk", "spoon_powder", "SIL"],
                ],
                Column.ACTION_ID: [[0, 2, 5, 1, 3, 0], [0, 1, 4, 0]],
                Column.COOKING_ACTIVITY: ["cereals", "milk"],
                Column.COOKING_ACTIVITY_ID: [0, 1],
                Column.END_TIME: [
                    [30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
                    [47.0, 215.0, 565.0, 747.0],
                ],
                Column.PERSON: ["P03", "P54"],
                Column.PERSON_ID: [0, 1],
                Column.SEQUENCE_LENGTH: [6, 4],
                Column.START_TIME: [
                    [1.0, 31.0, 151.0, 429.0, 576.0, 706.0],
                    [1.0, 48.0, 216.0, 566.0],
                ],
            },
            schema={
                Column.ACTION: pl.List(pl.String),
                Column.ACTION_ID: pl.List(pl.Int64),
                Column.COOKING_ACTIVITY: pl.String,
                Column.COOKING_ACTIVITY_ID: pl.Int64,
                Column.END_TIME: pl.List(pl.Float32),
                Column.PERSON: pl.String,
                Column.PERSON_ID: pl.Int64,
                Column.SEQUENCE_LENGTH: pl.UInt32,
                Column.START_TIME: pl.List(pl.Float32),
            },
        ),
    )


def test_group_by_sequence_empty(data_prepared_empty: pl.DataFrame) -> None:
    assert_frame_equal(
        group_by_sequence(data_prepared_empty),
        pl.DataFrame(
            {
                Column.ACTION: [],
                Column.ACTION_ID: [],
                Column.COOKING_ACTIVITY: [],
                Column.COOKING_ACTIVITY_ID: [],
                Column.END_TIME: [],
                Column.PERSON: [],
                Column.PERSON_ID: [],
                Column.SEQUENCE_LENGTH: [],
                Column.START_TIME: [],
            },
            schema={
                Column.ACTION: pl.List(pl.String),
                Column.ACTION_ID: pl.List(pl.Int64),
                Column.COOKING_ACTIVITY: pl.String,
                Column.COOKING_ACTIVITY_ID: pl.Int64,
                Column.END_TIME: pl.List(pl.Float32),
                Column.PERSON: pl.String,
                Column.PERSON_ID: pl.Int64,
                Column.SEQUENCE_LENGTH: pl.UInt32,
                Column.START_TIME: pl.List(pl.Float32),
            },
        ),
    )


##############################
#     Tests for to_array     #
##############################


def test_to_array(data_prepared: pl.DataFrame) -> None:
    mask = np.array(
        [[False, False, False, False, False, False], [False, False, False, False, True, True]],
        dtype=bool,
    )
    assert objects_are_equal(
        to_array(data_prepared),
        {
            Column.ACTION: np.ma.masked_array(
                data=np.array(
                    [
                        ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"],
                        ["SIL", "pour_milk", "spoon_powder", "SIL", "N/A", "N/A"],
                    ],
                    dtype=str,
                ),
                mask=mask,
            ),
            Column.ACTION_ID: np.ma.masked_array(
                data=np.array([[0, 2, 5, 1, 3, 0], [0, 1, 4, 0, 0, 0]]),
                mask=mask,
            ),
            Column.COOKING_ACTIVITY: np.array(["cereals", "milk"]),
            Column.COOKING_ACTIVITY_ID: np.array([0, 1]),
            Column.END_TIME: np.ma.masked_array(
                data=np.array(
                    [
                        [30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
                        [47.0, 215.0, 565.0, 747.0, 0.0, 0.0],
                    ]
                ),
                mask=mask,
            ),
            Column.PERSON: np.array(["P03", "P54"]),
            Column.PERSON_ID: np.array([0, 1]),
            Column.START_TIME: np.ma.masked_array(
                data=np.array(
                    [[1.0, 31.0, 151.0, 429.0, 576.0, 706.0], [1.0, 48.0, 216.0, 566.0, 0.0, 0.0]]
                ),
                mask=mask,
            ),
            Column.SEQUENCE_LENGTH: np.array([6, 4]),
        },
    )


def test_to_array_empty(data_prepared_empty: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_array(data_prepared_empty),
        {
            Column.ACTION: np.ma.masked_array(data=np.zeros(shape=(0, 0), dtype=str), mask=None),
            Column.ACTION_ID: np.ma.masked_array(data=np.zeros(shape=(0, 0), dtype=int), mask=None),
            Column.COOKING_ACTIVITY: np.array([], dtype=str),
            Column.COOKING_ACTIVITY_ID: np.array([], dtype=int),
            Column.END_TIME: np.ma.masked_array(
                data=np.zeros(shape=(0, 0), dtype=float), mask=None
            ),
            Column.PERSON: np.array([], dtype=str),
            Column.PERSON_ID: np.array([], dtype=int),
            Column.SEQUENCE_LENGTH: np.array([], dtype=int),
            Column.START_TIME: np.ma.masked_array(
                data=np.zeros(shape=(0, 0), dtype=float), mask=None
            ),
        },
    )


#############################
#     Tests for to_list     #
#############################


def test_to_list(data_prepared: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_list(data_prepared),
        {
            Column.ACTION: [
                ["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "SIL"],
                ["SIL", "pour_milk", "spoon_powder", "SIL"],
            ],
            Column.ACTION_ID: [[0, 2, 5, 1, 3, 0], [0, 1, 4, 0]],
            Column.COOKING_ACTIVITY: ["cereals", "milk"],
            Column.COOKING_ACTIVITY_ID: [0, 1],
            Column.END_TIME: [
                [30.0, 150.0, 428.0, 575.0, 705.0, 836.0],
                [47.0, 215.0, 565.0, 747.0],
            ],
            Column.PERSON: ["P03", "P54"],
            Column.PERSON_ID: [0, 1],
            Column.SEQUENCE_LENGTH: [6, 4],
            Column.START_TIME: [
                [1.0, 31.0, 151.0, 429.0, 576.0, 706.0],
                [1.0, 48.0, 216.0, 566.0],
            ],
        },
    )


def test_to_list_empty(data_prepared_empty: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_list(data_prepared_empty),
        {
            Column.ACTION: [],
            Column.ACTION_ID: [],
            Column.COOKING_ACTIVITY: [],
            Column.COOKING_ACTIVITY_ID: [],
            Column.END_TIME: [],
            Column.PERSON: [],
            Column.PERSON_ID: [],
            Column.SEQUENCE_LENGTH: [],
            Column.START_TIME: [],
        },
    )
