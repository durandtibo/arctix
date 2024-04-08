from __future__ import annotations

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
    download_annotations,
    load_annotation,
    load_annotations,
    parse_action_annotation_lines,
)


@pytest.fixture(scope="module")
def annotation_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
def annotation_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("dataset")
    save_text(
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("P03_cam01_P03_cereals.txt"),
    )
    save_text(  # duplicate example
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("P03_cam02_P03_cereals.txt"),
    )
    save_text(
        "1-47 SIL  \n48-215 pour_milk  \n216-565 spoon_powder  \n566-747 SIL  \n",
        path.joinpath("milk/P54_webcam02_P54_milk.txt"),
    )
    return path


##########################################
#     Tests for download_annotations     #
##########################################


def test_download_annotations(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_annotations(tmp_path)
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


def test_download_annotations_dir_exists(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.Path.is_dir", lambda *args, **kwargs: True),  # noqa: ARG005
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_annotations(tmp_path)
        download_mock.assert_not_called()
        tarfile_mock.assert_not_called()


def test_download_annotations_dir_exists_force_download(tmp_path: Path) -> None:
    with (
        patch("arctix.dataset.breakfast.Path.is_dir", lambda *args, **kwargs: True),  # noqa: ARG005
        patch("arctix.dataset.breakfast.download_drive_file") as download_mock,
        patch("arctix.dataset.breakfast.tarfile.open") as tarfile_mock,
    ):
        download_annotations(tmp_path, force_download=True)
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


##########################################
#     Tests for load_annotations     #
##########################################


def test_load_annotations_empty(tmp_path: Path) -> None:
    assert_frame_equal(load_annotations(tmp_path), pl.DataFrame({}))


def test_load_annotations(annotation_dir: Path) -> None:
    assert_frame_equal(
        load_annotations(annotation_dir),
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
                Column.PERSON_ID: [
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


def test_load_annotations_keep_duplicates(annotation_dir: Path) -> None:
    assert_frame_equal(
        load_annotations(annotation_dir, remove_duplicate=False),
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
                Column.PERSON_ID: [
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


#####################################
#     Tests for load_annotation     #
#####################################


def test_load_annotation_incorrect_extension() -> None:
    with pytest.raises(ValueError, match="Incorrect file extension."):
        load_annotation(Mock(spec=Path))


def test_load_annotation(annotation_file: Path) -> None:
    assert objects_are_equal(
        load_annotation(annotation_file),
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
            Column.PERSON_ID: ["P03", "P03", "P03", "P03", "P03", "P03"],
        },
    )


###################################################
#     Tests for parse_action_annotation_lines     #
###################################################


def test_parse_action_annotation_lines_empty() -> None:
    assert objects_are_equal(
        parse_action_annotation_lines([]),
        {
            Column.ACTION: [],
            Column.START_TIME: [],
            Column.END_TIME: [],
        },
    )


def test_parse_annotation_lines() -> None:
    assert objects_are_equal(
        parse_action_annotation_lines(
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
