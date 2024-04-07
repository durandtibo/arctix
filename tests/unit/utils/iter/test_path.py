from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iden.io import save_text

from arctix.utils.iter import PathLister

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    save_text("", path.joinpath("file.txt"))
    save_text("", path.joinpath("dir/file.txt"))
    return path


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
