from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from zipfile import ZipFile

import pytest
from iden.io import save_text

from arctix.dataset.epic_kitchen_100 import (
    ANNOTATION_FILENAMES,
    ANNOTATION_URL,
    download_data,
    is_annotation_path_ready,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def data_zip_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("epic-kitchens-100.zip.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        for filename in ANNOTATION_FILENAMES:
            zfile.writestr(f"epic-kitchens-100-annotations-master/{filename}", "")
    return path


###################################
#     Tests for download_data     #
###################################


@pytest.mark.parametrize("force_download", [True, False])
def test_download_data(data_zip_file: Path, tmp_path: Path, force_download: bool) -> None:
    with (
        patch("arctix.dataset.epic_kitchen_100.download_url_to_file") as download_mock,
        patch(
            "arctix.dataset.epic_kitchen_100.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
    ):
        download_data(tmp_path, force_download=force_download)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(tmp_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


def test_download_data_already_exists_force_download_false(tmp_path: Path) -> None:
    with patch(
        "arctix.dataset.epic_kitchen_100.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        download_data(tmp_path)
        # The file should not exist because the download step is skipped
        assert not tmp_path.joinpath("EPIC_100_train.csv").is_file()


def test_download_data_already_exists_force_download_true(
    data_zip_file: Path, tmp_path: Path
) -> None:
    with (
        patch(
            "arctix.dataset.epic_kitchen_100.is_annotation_path_ready",
            Mock(return_value=True),
        ),
        patch(
            "arctix.dataset.epic_kitchen_100.TemporaryDirectory.__enter__",
            Mock(return_value=data_zip_file.parent),
        ),
        patch("arctix.dataset.epic_kitchen_100.download_url_to_file") as download_mock,
    ):
        download_data(tmp_path, force_download=True)
        download_mock.assert_called_once_with(
            ANNOTATION_URL, data_zip_file.as_posix(), progress=True
        )
        assert all(tmp_path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


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
