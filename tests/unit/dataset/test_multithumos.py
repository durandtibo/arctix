from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from iden.io import save_text
from polars.testing import assert_frame_equal

from arctix.dataset.multithumos import (
    ANNOTATION_FILENAMES,
    ANNOTATION_URL,
    Column,
    MetadataKeys,
    download_data,
    fetch_data,
    filter_by_split,
    generate_split_column,
    group_by_sequence,
    is_annotation_path_ready,
    load_annotation_file,
    load_data,
    parse_annotation_lines,
    prepare_data,
    to_array,
    to_list,
)
from arctix.utils.vocab import Vocabulary


@pytest.fixture(scope="module")
def data_zip_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("multithumos.zip.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        for filename in ANNOTATION_FILENAMES:
            zfile.writestr(f"multithumos/{filename}", "")
    return path


@pytest.fixture(scope="module")
def data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("annotations").joinpath("dribble.txt")
    save_text(
        "\n".join(
            [
                "",
                "  video_validation_0000266 72.00 76.00  ",
                " video_validation_0000681 44.00 50.00 ",
                "video_validation_0000682 1.00 5.00",
                "   ",
                "video_validation_0000682 79.00 83.00",
            ]
        ),
        path,
    )
    return path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("dataset")
    save_text(
        "\n".join(
            [
                "",
                "  video_validation_0000266 72.00 76.00  ",
                " video_validation_0000681 44.00 50.00 ",
                "video_validation_0000682 1.00 5.00",
                "   ",
                "video_validation_0000682 79.00 83.00",
            ]
        ),
        path.joinpath("annotations").joinpath("dribble.txt"),
    )
    save_text(
        "\n".join(
            [
                "video_validation_0000682 17.00 18.00",
                "video_validation_0000902 2.00 3.00",
                "video_validation_0000902 4.00 5.00",
                "video_validation_0000902 20.00 20.00",
                "video_validation_0000682 17.00 18.00",
            ]
        ),
        path.joinpath("annotations").joinpath("guard.txt"),
    )
    return path


@pytest.fixture
def data_raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION: [
                "dribble",
                "dribble",
                "dribble",
                "guard",
                "guard",
                "dribble",
                "guard",
                "guard",
                "guard",
            ],
            Column.END_TIME: [76.0, 50.0, 5.0, 18.0, 18.0, 83.0, 3.0, 5.0, 20.0],
            Column.START_TIME: [72.0, 44.0, 1.0, 17.0, 17.0, 79.0, 2.0, 4.0, 20.0],
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000902",
                "video_validation_0000902",
                "video_validation_0000902",
            ],
        },
        schema={
            Column.ACTION: pl.String,
            Column.END_TIME: pl.Float64,
            Column.START_TIME: pl.Float64,
            Column.VIDEO: pl.String,
        },
    )


@pytest.fixture
def data_prepared() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.ACTION: [
                "dribble",
                "dribble",
                "dribble",
                "guard",
                "guard",
                "dribble",
                "guard",
                "guard",
                "guard",
            ],
            Column.ACTION_ID: [1, 1, 1, 0, 0, 1, 0, 0, 0],
            Column.END_TIME: [76.0, 50.0, 5.0, 18.0, 18.0, 83.0, 3.0, 5.0, 20.0],
            Column.SPLIT: [
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
            ],
            Column.START_TIME: [72.0, 44.0, 1.0, 17.0, 17.0, 79.0, 2.0, 4.0, 20.0],
            Column.START_TIME_DIFF: [0.0, 0.0, 0.0, 16.0, 0.0, 62.0, 0.0, 2.0, 16.0],
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000682",
                "video_validation_0000902",
                "video_validation_0000902",
                "video_validation_0000902",
            ],
        },
        schema={
            Column.ACTION: pl.String,
            Column.ACTION_ID: pl.Int64,
            Column.END_TIME: pl.Float64,
            Column.SPLIT: pl.String,
            Column.START_TIME: pl.Float64,
            Column.START_TIME_DIFF: pl.Float64,
            Column.VIDEO: pl.String,
        },
    )


@pytest.fixture
def vocab_action() -> Vocabulary:
    return Vocabulary(Counter({"guard": 5, "dribble": 4}))


################################
#     Tests for fetch_data     #
################################


def test_fetch_data(data_dir: Path, data_raw: pl.DataFrame) -> None:
    with patch("arctix.dataset.multithumos.download_data") as download_mock:
        data = fetch_data(data_dir)
        download_mock.assert_called_once_with(data_dir, False)
    assert_frame_equal(data, data_raw)


###################################
#     Tests for download_data     #
###################################


@pytest.mark.parametrize("force_download", [True, False])
def test_download_data(data_zip_file: Path, tmp_path: Path, force_download: bool) -> None:
    data_path = tmp_path.joinpath("multithumos")
    with (
        patch("arctix.dataset.multithumos.download_url_to_file") as download_mock,
        patch(
            "arctix.dataset.multithumos.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
    ):
        download_data(data_path, force_download=force_download)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(data_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def test_download_data_already_exists_force_download_false(tmp_path: Path) -> None:
    with patch(
        "arctix.dataset.multithumos.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        download_data(tmp_path)
        # The file should not exist because the download step is skipped
        assert not any(tmp_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def test_download_data_already_exists_force_download_true(
    data_zip_file: Path, tmp_path: Path
) -> None:
    data_path = tmp_path.joinpath("multithumos")
    with (
        patch(
            "arctix.dataset.multithumos.is_annotation_path_ready",
            Mock(return_value=True),
        ),
        patch(
            "arctix.dataset.multithumos.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
        patch("arctix.dataset.multithumos.download_url_to_file") as download_mock,
    ):
        download_data(data_path, force_download=True)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(data_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


##############################################
#     Tests for is_annotation_path_ready     #
##############################################


def test_is_annotation_path_ready_true(tmp_path: Path) -> None:
    for filename in ANNOTATION_FILENAMES:
        save_text("", tmp_path.joinpath(filename))
    assert is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_all(tmp_path: Path) -> None:
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_partial(tmp_path: Path) -> None:
    for filename in ANNOTATION_FILENAMES[::2]:
        save_text("", tmp_path.joinpath(filename))
    assert not is_annotation_path_ready(tmp_path)


###############################
#     Tests for load_data     #
###############################


def test_load_data(data_dir: Path, data_raw: pl.DataFrame) -> None:
    assert_frame_equal(load_data(data_dir), data_raw)


##########################################
#     Tests for load_annotation_file     #
##########################################


def test_load_annotation_file_incorrect_extension() -> None:
    with pytest.raises(ValueError, match=r"Incorrect file extension."):
        load_annotation_file(Mock(spec=Path))


def test_load_annotation_file(data_file: Path) -> None:
    assert objects_are_equal(
        load_annotation_file(data_file),
        {
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000682",
            ],
            Column.START_TIME: [72.0, 44.0, 1.0, 79.0],
            Column.END_TIME: [76.0, 50.0, 5.0, 83.0],
            Column.ACTION: ["dribble", "dribble", "dribble", "dribble"],
        },
    )


############################################
#     Tests for parse_annotation_lines     #
############################################


def test_parse_annotation_lines() -> None:
    assert objects_are_equal(
        parse_annotation_lines(
            [
                "  video_validation_0000266 72.80 76.40  ",
                " video_validation_0000681 44.00 50.90 ",
                "video_validation_0000682 1.50 5.40",
                "   ",
                "video_validation_0000682 79.30 83.90",
            ]
        ),
        {
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000682",
            ],
            Column.START_TIME: [72.80, 44.00, 1.50, 79.30],
            Column.END_TIME: [76.40, 50.90, 5.40, 83.90],
        },
    )


def test_parse_annotation_lines_empty() -> None:
    assert objects_are_equal(
        parse_annotation_lines([]),
        {Column.VIDEO: [], Column.START_TIME: [], Column.END_TIME: []},
    )


##################################
#     Tests for prepare_data     #
##################################


def test_prepare_data(
    data_raw: pl.DataFrame, data_prepared: pl.DataFrame, vocab_action: Vocabulary
) -> None:
    data, metadata = prepare_data(data_raw)
    assert_frame_equal(data, data_prepared)
    assert objects_are_equal(metadata, {MetadataKeys.VOCAB_ACTION: vocab_action})


def test_prepare_data_empty() -> None:
    data, metadata = prepare_data(
        pl.DataFrame(
            {
                Column.ACTION: [],
                Column.END_TIME: [],
                Column.START_TIME: [],
                Column.VIDEO: [],
            },
            schema={
                Column.ACTION: pl.String,
                Column.END_TIME: pl.Float64,
                Column.START_TIME: pl.Float64,
                Column.VIDEO: pl.String,
            },
        ),
    )
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION: [],
                Column.ACTION_ID: [],
                Column.END_TIME: [],
                Column.SPLIT: [],
                Column.START_TIME: [],
                Column.START_TIME_DIFF: [],
                Column.VIDEO: [],
            },
            schema={
                Column.ACTION: pl.String,
                Column.ACTION_ID: pl.Int64,
                Column.END_TIME: pl.Float64,
                Column.SPLIT: pl.String,
                Column.START_TIME: pl.Float64,
                Column.START_TIME_DIFF: pl.Float64,
                Column.VIDEO: pl.String,
            },
        ),
    )
    assert objects_are_equal(metadata, {MetadataKeys.VOCAB_ACTION: Vocabulary(Counter({}))})


def test_prepare_data_split_validation() -> None:
    data, metadata = prepare_data(
        pl.DataFrame(
            {
                Column.ACTION: [
                    "dribble",
                    "dribble",
                    "dribble",
                    "guard",
                    "dribble",
                    "guard",
                    "guard",
                    "guard",
                ],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 83.90, 3.60, 5.07, 20.49],
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 79.30, 2.97, 4.54, 20.22],
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000902",
                    "video_validation_0000902",
                    "video_test_0000902",
                ],
            },
            schema={
                Column.ACTION: pl.String,
                Column.END_TIME: pl.Float64,
                Column.START_TIME: pl.Float64,
                Column.VIDEO: pl.String,
            },
        ),
        split="validation",
    )
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.ACTION: [
                    "dribble",
                    "dribble",
                    "dribble",
                    "guard",
                    "dribble",
                    "guard",
                    "guard",
                ],
                Column.ACTION_ID: [1, 1, 1, 0, 1, 0, 0],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 83.90, 3.60, 5.07],
                Column.SPLIT: [
                    "validation",
                    "validation",
                    "validation",
                    "validation",
                    "validation",
                    "validation",
                    "validation",
                ],
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 79.30, 2.97, 4.54],
                Column.START_TIME_DIFF: [0.0, 0.0, 0.0, 16.07, 61.73, 0.0, 1.57],
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000902",
                    "video_validation_0000902",
                ],
            },
            schema={
                Column.ACTION: pl.String,
                Column.ACTION_ID: pl.Int64,
                Column.END_TIME: pl.Float64,
                Column.SPLIT: pl.String,
                Column.START_TIME: pl.Float64,
                Column.START_TIME_DIFF: pl.Float64,
                Column.VIDEO: pl.String,
            },
        ),
    )
    assert objects_are_equal(
        metadata,
        {MetadataKeys.VOCAB_ACTION: Vocabulary(Counter({"guard": 4, "dribble": 4}))},
    )


###########################################
#     Tests for generate_split_column     #
###########################################


def test_generate_split_column() -> None:
    assert_frame_equal(
        generate_split_column(
            pl.DataFrame(
                {
                    Column.VIDEO: [
                        "video_validation_0000266",
                        "video_test_0000862",
                        "video_validation_0000681",
                        "video_validation_0000682",
                        "video_test_0000234",
                        "video_validation_0000902",
                    ],
                    "col": [1, 2, 3, 4, 5, 6],
                },
                schema={Column.VIDEO: pl.String, "col": pl.Int64},
            )
        ),
        pl.DataFrame(
            {
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_test_0000862",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_test_0000234",
                    "video_validation_0000902",
                ],
                "col": [1, 2, 3, 4, 5, 6],
                Column.SPLIT: [
                    "validation",
                    "test",
                    "validation",
                    "validation",
                    "test",
                    "validation",
                ],
            },
            schema={Column.VIDEO: pl.String, "col": pl.Int64, Column.SPLIT: pl.String},
        ),
    )


def test_generate_split_column_empty() -> None:
    assert_frame_equal(
        generate_split_column(
            pl.DataFrame(
                {Column.VIDEO: [], "col": []},
                schema={Column.VIDEO: pl.String, "col": pl.Int64},
            )
        ),
        pl.DataFrame(
            {
                Column.VIDEO: [],
                "col": [],
                Column.SPLIT: [],
            },
            schema={Column.VIDEO: pl.String, "col": pl.Int64, Column.SPLIT: pl.String},
        ),
    )


#####################################
#     Tests for filter_by_split     #
#####################################


def test_filter_by_split_all() -> None:
    assert_frame_equal(
        filter_by_split(
            pl.DataFrame(
                {
                    Column.SPLIT: [
                        "validation",
                        "test",
                        "test",
                        "validation",
                        "validation",
                        "test",
                        "test",
                        "",
                    ],
                    "col": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            ),
        ),
        pl.DataFrame(
            {
                Column.SPLIT: [
                    "validation",
                    "test",
                    "test",
                    "validation",
                    "validation",
                    "test",
                    "test",
                ],
                "col": [1, 2, 3, 4, 5, 6, 7],
            }
        ),
    )


def test_filter_by_split_validation() -> None:
    assert_frame_equal(
        filter_by_split(
            pl.DataFrame(
                {
                    Column.SPLIT: [
                        "validation",
                        "test",
                        "test",
                        "validation",
                        "validation",
                        "test",
                        "test",
                        "test",
                    ],
                    "col": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            ),
            split="validation",
        ),
        pl.DataFrame(
            {
                Column.SPLIT: ["validation", "validation", "validation"],
                "col": [1, 4, 5],
            }
        ),
    )


def test_filter_by_split_test() -> None:
    assert_frame_equal(
        filter_by_split(
            pl.DataFrame(
                {
                    Column.SPLIT: [
                        "validation",
                        "test",
                        "test",
                        "validation",
                        "validation",
                        "test",
                        "test",
                        "test",
                    ],
                    "col": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            ),
            split="test",
        ),
        pl.DataFrame(
            {
                Column.SPLIT: ["test", "test", "test", "test", "test"],
                "col": [2, 3, 6, 7, 8],
            }
        ),
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
                    ["dribble"],
                    ["dribble"],
                    ["dribble", "guard", "guard", "dribble"],
                    ["guard", "guard", "guard"],
                ],
                Column.ACTION_ID: [[1], [1], [1, 0, 0, 1], [0, 0, 0]],
                Column.END_TIME: [
                    [76.0],
                    [50.0],
                    [5.0, 18.0, 18.0, 83.0],
                    [3.0, 5.0, 20.0],
                ],
                Column.SEQUENCE_LENGTH: [1, 1, 4, 3],
                Column.SPLIT: ["validation", "validation", "validation", "validation"],
                Column.START_TIME: [
                    [72.0],
                    [44.0],
                    [1.0, 17.0, 17.0, 79.0],
                    [2.0, 4.0, 20.0],
                ],
                Column.START_TIME_DIFF: [
                    [0.0],
                    [0.0],
                    [0.0, 16.0, 0.0, 62.0],
                    [0.0, 2.0, 16.0],
                ],
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_validation_0000902",
                ],
            },
            schema={
                Column.ACTION: pl.List(pl.String),
                Column.ACTION_ID: pl.List(pl.Int64),
                Column.END_TIME: pl.List(pl.Float64),
                Column.SEQUENCE_LENGTH: pl.Int64,
                Column.SPLIT: pl.String,
                Column.START_TIME: pl.List(pl.Float64),
                Column.START_TIME_DIFF: pl.List(pl.Float64),
                Column.VIDEO: pl.String,
            },
        ),
    )


def test_group_by_sequence_empty() -> None:
    assert_frame_equal(
        group_by_sequence(
            pl.DataFrame(
                {
                    Column.ACTION: [],
                    Column.ACTION_ID: [],
                    Column.END_TIME: [],
                    Column.SPLIT: [],
                    Column.START_TIME: [],
                    Column.START_TIME_DIFF: [],
                    Column.VIDEO: [],
                },
                schema={
                    Column.ACTION: pl.String,
                    Column.ACTION_ID: pl.Int64,
                    Column.END_TIME: pl.Float64,
                    Column.SPLIT: pl.String,
                    Column.START_TIME: pl.Float64,
                    Column.START_TIME_DIFF: pl.Float64,
                    Column.VIDEO: pl.String,
                },
            )
        ),
        pl.DataFrame(
            {
                Column.ACTION: [],
                Column.ACTION_ID: [],
                Column.END_TIME: [],
                Column.SEQUENCE_LENGTH: [],
                Column.SPLIT: [],
                Column.START_TIME: [],
                Column.START_TIME_DIFF: [],
                Column.VIDEO: [],
            },
            schema={
                Column.ACTION: pl.List(pl.String),
                Column.ACTION_ID: pl.List(pl.Int64),
                Column.END_TIME: pl.List(pl.Float64),
                Column.SEQUENCE_LENGTH: pl.Int64,
                Column.SPLIT: pl.String,
                Column.START_TIME: pl.List(pl.Float64),
                Column.START_TIME_DIFF: pl.List(pl.Float64),
                Column.VIDEO: pl.String,
            },
        ),
    )


##############################
#     Tests for to_array     #
##############################


def test_to_array(data_prepared: pl.DataFrame) -> None:
    mask = np.array(
        [
            [False, True, True, True],
            [False, True, True, True],
            [False, False, False, False],
            [False, False, False, True],
        ]
    )
    assert objects_are_equal(
        to_array(data_prepared),
        {
            Column.ACTION: np.ma.masked_array(
                data=np.array(
                    [
                        ["dribble", "N/A", "N/A", "N/A"],
                        ["dribble", "N/A", "N/A", "N/A"],
                        ["dribble", "guard", "guard", "dribble"],
                        ["guard", "guard", "guard", "N/A"],
                    ],
                    dtype=str,
                ),
                mask=mask,
            ),
            Column.ACTION_ID: np.ma.masked_array(
                data=np.array(
                    [[1, -1, -1, -1], [1, -1, -1, -1], [1, 0, 0, 1], [0, 0, 0, -1]], dtype=int
                ),
                mask=mask,
            ),
            Column.END_TIME: np.ma.masked_array(
                data=np.array(
                    [
                        [76.0, -1.0, -1.0, -1.0],
                        [50.0, -1.0, -1.0, -1.0],
                        [5.0, 18.0, 18.0, 83.0],
                        [3.0, 5.0, 20.0, -1.0],
                    ],
                    dtype=float,
                ),
                mask=mask,
            ),
            Column.SEQUENCE_LENGTH: np.array([1, 1, 4, 3], dtype=int),
            Column.SPLIT: np.array(
                ["validation", "validation", "validation", "validation"], dtype=str
            ),
            Column.START_TIME: np.ma.masked_array(
                data=np.array(
                    [
                        [72.0, -1.0, -1.0, -1.0],
                        [44.0, -1.0, -1.0, -1.0],
                        [1.0, 17.0, 17.0, 79.0],
                        [2.0, 4.0, 20.0, -1.0],
                    ],
                    dtype=float,
                ),
                mask=mask,
            ),
            Column.START_TIME_DIFF: np.ma.masked_array(
                data=np.array(
                    [
                        [0.0, -1.0, -1.0, -1.0],
                        [0.0, -1.0, -1.0, -1.0],
                        [0.0, 16.0, 0.0, 62.0],
                        [0.0, 2.0, 16.0, -1.0],
                    ],
                    dtype=float,
                ),
                mask=mask,
            ),
        },
    )


def test_to_array_empty() -> None:
    mask = None
    assert objects_are_equal(
        to_array(
            pl.DataFrame(
                {
                    Column.ACTION: [],
                    Column.ACTION_ID: [],
                    Column.END_TIME: [],
                    Column.SPLIT: [],
                    Column.START_TIME: [],
                    Column.START_TIME_DIFF: [],
                    Column.VIDEO: [],
                },
                schema={
                    Column.ACTION: pl.String,
                    Column.ACTION_ID: pl.Int64,
                    Column.END_TIME: pl.Float64,
                    Column.SPLIT: pl.String,
                    Column.START_TIME: pl.Float64,
                    Column.START_TIME_DIFF: pl.Float64,
                    Column.VIDEO: pl.String,
                },
            )
        ),
        {
            Column.ACTION: np.ma.masked_array(data=np.zeros(shape=(0, 0), dtype=str), mask=mask),
            Column.ACTION_ID: np.ma.masked_array(data=np.zeros(shape=(0, 0), dtype=int), mask=mask),
            Column.END_TIME: np.ma.masked_array(
                data=np.zeros(shape=(0, 0), dtype=float), mask=mask
            ),
            Column.SEQUENCE_LENGTH: np.array([], dtype=int),
            Column.SPLIT: np.array([], dtype=str),
            Column.START_TIME: np.ma.masked_array(
                data=np.zeros(shape=(0, 0), dtype=float), mask=mask
            ),
            Column.START_TIME_DIFF: np.ma.masked_array(
                data=np.zeros(shape=(0, 0), dtype=float), mask=mask
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
                ["dribble"],
                ["dribble"],
                ["dribble", "guard", "guard", "dribble"],
                ["guard", "guard", "guard"],
            ],
            Column.ACTION_ID: [[1], [1], [1, 0, 0, 1], [0, 0, 0]],
            Column.END_TIME: [
                [76.0],
                [50.0],
                [5.0, 18.0, 18.0, 83.0],
                [3.0, 5.0, 20.0],
            ],
            Column.SEQUENCE_LENGTH: [1, 1, 4, 3],
            Column.SPLIT: ["validation", "validation", "validation", "validation"],
            Column.START_TIME: [
                [72.0],
                [44.0],
                [1.0, 17.0, 17.0, 79.0],
                [2.0, 4.0, 20.0],
            ],
            Column.START_TIME_DIFF: [
                [0.0],
                [0.0],
                [0.0, 16.0, 0.0, 62.0],
                [0.0, 2.0, 16.0],
            ],
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000902",
            ],
        },
    )


def test_to_list_empty() -> None:
    assert objects_are_equal(
        to_list(
            pl.DataFrame(
                {
                    Column.ACTION: [],
                    Column.ACTION_ID: [],
                    Column.END_TIME: [],
                    Column.SPLIT: [],
                    Column.START_TIME: [],
                    Column.START_TIME_DIFF: [],
                    Column.VIDEO: [],
                },
                schema={
                    Column.ACTION: pl.String,
                    Column.ACTION_ID: pl.Int64,
                    Column.END_TIME: pl.Float64,
                    Column.SPLIT: pl.String,
                    Column.START_TIME: pl.Float64,
                    Column.START_TIME_DIFF: pl.Float64,
                    Column.VIDEO: pl.String,
                },
            )
        ),
        {
            Column.ACTION: [],
            Column.ACTION_ID: [],
            Column.END_TIME: [],
            Column.SEQUENCE_LENGTH: [],
            Column.SPLIT: [],
            Column.START_TIME: [],
            Column.START_TIME_DIFF: [],
            Column.VIDEO: [],
        },
    )
