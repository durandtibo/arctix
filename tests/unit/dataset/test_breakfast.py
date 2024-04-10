from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import Mock, call, patch

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
    load_annotation_file,
    load_data,
    parse_annotation_lines,
    prepare_data,
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


################################
#     Tests for fetch_data     #
################################


def test_fetch_data_remove_duplicate_examples(data_dir: Path) -> None:
    with patch("arctix.dataset.breakfast.download_data") as download_mock:
        data = fetch_data(data_dir, name="segmentation_coarse")
        download_mock.assert_called_once_with(data_dir, False)
    assert_frame_equal(
        data,
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
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )


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
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
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
                Column.START_TIME: [
                    1.0,
                    31.0,
                    151.0,
                    429.0,
                    576.0,
                    706.0,
                    1.0,
                    31.0,
                    151.0,
                    429.0,
                    576.0,
                    706.0,
                    1.0,
                    48.0,
                    216.0,
                    566.0,
                ],
                Column.END_TIME: [
                    30.0,
                    150.0,
                    428.0,
                    575.0,
                    705.0,
                    836.0,
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


def test_load_data_empty(tmp_path: Path) -> None:
    assert_frame_equal(load_data(tmp_path), pl.DataFrame({}))


def test_load_data(data_dir: Path) -> None:
    assert_frame_equal(
        load_data(data_dir),
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
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )


def test_load_data_keep_duplicates(data_dir: Path) -> None:
    assert_frame_equal(
        load_data(data_dir, remove_duplicate=False),
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
                Column.START_TIME: [
                    1.0,
                    31.0,
                    151.0,
                    429.0,
                    576.0,
                    706.0,
                    1.0,
                    31.0,
                    151.0,
                    429.0,
                    576.0,
                    706.0,
                    1.0,
                    48.0,
                    216.0,
                    566.0,
                ],
                Column.END_TIME: [
                    30.0,
                    150.0,
                    428.0,
                    575.0,
                    705.0,
                    836.0,
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


##################################
#     Tests for prepare_data     #
##################################


def test_prepare_data_empty() -> None:
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
            }
        )
    )
    assert_frame_equal(
        data,
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
                Column.ACTION_ID: [0, 2, 3, 1, 4, 0, 0, 1, 5, 0],
                Column.PERSON_ID: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
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
                        "pour_cereals": 1,
                        "stir_cereals": 1,
                        "spoon_powder": 1,
                    }
                )
            ),
            "vocab_person": Vocabulary(Counter({"P03": 6, "P54": 4})),
        },
    )
