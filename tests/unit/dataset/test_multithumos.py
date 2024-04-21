from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from zipfile import ZipFile

import pytest
from coola import objects_are_equal
from iden.io import save_text

from arctix.dataset.multithumos import (
    ANNOTATION_URL,
    Column,
    download_data,
    is_annotation_path_ready,
    parse_annotation_lines,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    path = tmp_path_factory.mktemp("data").joinpath("action.txt")
    save_text(
        (
            "video_validation_0000266 72.80 76.40  \n",
            "video_validation_0000681 44.00 50.90 \n",
            "video_validation_0000682 1.50 5.40\n",
            "video_validation_0000682 79.30 83.90\n",
        ),
        path,
    )
    return path


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
