from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from zipfile import ZipFile

import pytest
from iden.io import save_text

from arctix.dataset.multithumos import (
    ANNOTATION_URL,
    download_data,
    is_annotation_path_ready,
)

if TYPE_CHECKING:
    from pathlib import Path

###################################
#     Tests for download_data     #
###################################


@pytest.fixture(scope="module")
def data_zip_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("multithumos.zip.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        zfile.writestr("multithumos/README", "")
        zfile.writestr("multithumos/class_list.txt", "")
        zfile.writestr("multithumos/annotations", "")
    return path


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
