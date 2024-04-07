from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iden.io import save_text

from arctix.utils.iter import DirFilter, FileFilter, PathLister

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    save_text("", path.joinpath("file.txt"))
    save_text("", path.joinpath("dir/file.txt"))
    return path


###############################
#     Tests for DirFilter     #
###############################


def test_dir_filter_repr() -> None:
    assert repr(DirFilter([])).startswith("DirFilter(")


def test_dir_filter_str() -> None:
    assert str(DirFilter([])).startswith("DirFilter(")


def test_dir_filter_iter(data_path: Path) -> None:
    assert list(
        DirFilter(
            [
                data_path.joinpath("dir/file.txt"),
                data_path.joinpath("dir/"),
                data_path.joinpath("file.txt"),
            ]
        )
    ) == [data_path.joinpath("dir/")]


def test_dir_filter_iter_empty() -> None:
    assert list(DirFilter([])) == []


################################
#     Tests for FileFilter     #
################################


def test_file_filter_repr() -> None:
    assert repr(FileFilter([])).startswith("FileFilter(")


def test_file_filter_str() -> None:
    assert str(FileFilter([])).startswith("FileFilter(")


def test_file_filter_iter(data_path: Path) -> None:
    assert list(
        FileFilter(
            [
                data_path.joinpath("dir/file.txt"),
                data_path.joinpath("dir/"),
                data_path.joinpath("file.txt"),
            ]
        )
    ) == [data_path.joinpath("dir/file.txt"), data_path.joinpath("file.txt")]


def test_dir_filter_iter_file() -> None:
    assert list(FileFilter([])) == []


################################
#     Tests for PathLister     #
################################


def test_path_lister_repr() -> None:
    assert repr(PathLister([])).startswith("PathLister(")


def test_path_lister_str() -> None:
    assert str(PathLister([])).startswith("PathLister(")


def test_path_lister_iter_empty(tmp_path: Path) -> None:
    assert list(PathLister([tmp_path])) == []


def test_path_lister_iter(data_path: Path) -> None:
    assert list(PathLister([data_path], pattern="*.txt")) == [data_path.joinpath("file.txt")]


def test_path_lister_iter_recursive(data_path: Path) -> None:
    assert list(PathLister([data_path], pattern="**/*.txt")) == [
        data_path.joinpath("dir/file.txt"),
        data_path.joinpath("file.txt"),
    ]


def test_path_lister_iter_deterministic_false(data_path: Path) -> None:
    assert set(PathLister([data_path], pattern="**/*.txt", deterministic=False)) == {
        data_path.joinpath("dir/file.txt"),
        data_path.joinpath("file.txt"),
    }
