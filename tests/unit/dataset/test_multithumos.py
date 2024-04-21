from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import polars as pl
import pytest
from coola import objects_are_equal
from iden.io import save_text
from polars.testing import assert_frame_equal

from arctix.dataset.multithumos import (
    ANNOTATION_URL,
    Column,
    download_data,
    fetch_data,
    is_annotation_path_ready,
    load_annotation_file,
    load_data,
    parse_annotation_lines,
)


@pytest.fixture(scope="module")
def data_zip_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("multithumos.zip.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        zfile.writestr("multithumos/README", "")
        zfile.writestr("multithumos/class_list.txt", "")
        zfile.writestr("multithumos/annotations", "")
    return path


@pytest.fixture(scope="module")
def data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("annotations").joinpath("dribble.txt")
    save_text(
        "\n".join(
            [
                "",
                "  video_validation_0000266 72.80 76.40  ",
                " video_validation_0000681 44.00 50.90 ",
                "video_validation_0000682 1.50 5.40",
                "   ",
                "video_validation_0000682 79.30 83.90",
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
                "  video_validation_0000266 72.80 76.40  ",
                " video_validation_0000681 44.00 50.90 ",
                "video_validation_0000682 1.50 5.40",
                "   ",
                "video_validation_0000682 79.30 83.90",
            ]
        ),
        path.joinpath("annotations").joinpath("dribble.txt"),
    )
    save_text(
        "\n".join(
            [
                "video_validation_0000682 17.57 18.33",
                "video_validation_0000902 2.97 3.60",
                "video_validation_0000902 4.54 5.07",
                "video_validation_0000902 20.22 20.49",
                "video_validation_0000682 17.57 18.33",
            ]
        ),
        path.joinpath("annotations").joinpath("guard.txt"),
    )
    return path


################################
#     Tests for fetch_data     #
################################


def test_fetch_data_remove_duplicate_examples(data_dir: Path) -> None:
    with patch("arctix.dataset.multithumos.download_data") as download_mock:
        data = fetch_data(data_dir)
        download_mock.assert_called_once_with(data_dir, False)
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000902",
                    "video_validation_0000902",
                    "video_validation_0000902",
                ],
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 79.30, 2.97, 4.54, 20.22],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 83.90, 3.60, 5.07, 20.49],
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
            },
        ),
    )


def test_fetch_data_keep_duplicate_examples(data_dir: Path) -> None:
    with patch("arctix.dataset.multithumos.download_data") as download_mock:
        data = fetch_data(data_dir, remove_duplicate=False, force_download=True)
        download_mock.assert_called_once_with(data_dir, True)
    assert_frame_equal(
        data,
        pl.DataFrame(
            {
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
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 17.57, 79.30, 2.97, 4.54, 20.22],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 18.33, 83.90, 3.60, 5.07, 20.49],
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
            },
        ),
    )


###################################
#     Tests for download_data     #
###################################


@pytest.mark.parametrize("force_download", [True, False])
def test_download_data(data_zip_file: Path, tmp_path: Path, force_download: bool) -> None:
    with (
        patch("arctix.dataset.multithumos.download_url_to_file") as download_mock,
        patch(
            "arctix.dataset.multithumos.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
    ):
        download_data(tmp_path.joinpath("multithumos"), force_download=force_download)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert tmp_path.joinpath("multithumos/README").is_file()
        assert tmp_path.joinpath("multithumos/class_list.txt").is_file()
        assert tmp_path.joinpath("multithumos/annotations").is_file()


def test_download_data_already_exists_force_download_false(tmp_path: Path) -> None:
    with patch(
        "arctix.dataset.multithumos.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        download_data(tmp_path)
        # The file should not exist because the download step is skipped
        assert not tmp_path.joinpath("multithumos/README").is_file()


def test_download_data_already_exists_force_download_true(
    data_zip_file: Path, tmp_path: Path
) -> None:
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
        download_data(tmp_path.joinpath("multithumos"), force_download=True)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert tmp_path.joinpath("multithumos/README").is_file()
        assert tmp_path.joinpath("multithumos/class_list.txt").is_file()
        assert tmp_path.joinpath("multithumos/annotations").is_file()


##############################################
#     Tests for is_annotation_path_ready     #
##############################################


def test_is_annotation_path_ready_true(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("class_list.txt"))
    for i in range(65):
        save_text("", tmp_path.joinpath(f"annotations/{i + 1}.txt"))
    assert is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_readme(tmp_path: Path) -> None:
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_class_list(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README.txt"))
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_annotations(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("class_list.txt"))
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_annotation_file(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("README.txt"))
    for i in range(64):
        save_text("", tmp_path.joinpath(f"annotations/{i + 1}.txt"))
    assert not is_annotation_path_ready(tmp_path)


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
                Column.VIDEO: [
                    "video_validation_0000266",
                    "video_validation_0000681",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000682",
                    "video_validation_0000902",
                    "video_validation_0000902",
                    "video_validation_0000902",
                ],
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 79.30, 2.97, 4.54, 20.22],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 83.90, 3.60, 5.07, 20.49],
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
            },
        ),
    )


def test_load_data_keep_duplicates(data_dir: Path) -> None:
    assert_frame_equal(
        load_data(data_dir, remove_duplicate=False),
        pl.DataFrame(
            {
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
                Column.START_TIME: [72.80, 44.00, 1.50, 17.57, 17.57, 79.30, 2.97, 4.54, 20.22],
                Column.END_TIME: [76.40, 50.90, 5.40, 18.33, 18.33, 83.90, 3.60, 5.07, 20.49],
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
            Column.VIDEO: [
                "video_validation_0000266",
                "video_validation_0000681",
                "video_validation_0000682",
                "video_validation_0000682",
            ],
            Column.START_TIME: [72.80, 44.00, 1.50, 79.30],
            Column.END_TIME: [76.40, 50.90, 5.40, 83.90],
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
