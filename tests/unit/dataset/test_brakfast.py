from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import call, patch

from arctix.dataset.breakfast import URLS, download_annotations

if TYPE_CHECKING:
    from pathlib import Path

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
