from __future__ import annotations

from typing import TYPE_CHECKING

from arctix.testing import gdown_available
from arctix.utils.download import download_drive_file

if TYPE_CHECKING:
    from pathlib import Path


TEST_URL = "https://docs.google.com/document/d/1PK1HGa3HViKSJhAhvQgZNEYB72J0DhcXPNKuSpI4N80"

#########################################
#     Tests for download_drive_file     #
#########################################


@gdown_available
def test_download_drive_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    assert not path.is_file()
    download_drive_file(TEST_URL, path, fuzzy=True)
    assert path.is_file()
