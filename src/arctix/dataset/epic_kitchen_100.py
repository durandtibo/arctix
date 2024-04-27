r"""Contain code to download and prepare the EPIC-KITCHENS-100 data.

The following documentation assumes the data are downloaded in the
directory `/path/to/data/epic_kitchen_100/`.
"""

from __future__ import annotations

__all__ = ["download_data"]

import logging
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from iden.utils.path import sanitize_path

from arctix.utils.download import download_url_to_file

logger = logging.getLogger(__name__)

ANNOTATION_URL = (
    "https://github.com/epic-kitchens/epic-kitchens-100-annotations/archive/refs/heads/master.zip"
)

ANNOTATION_FILENAMES = [
    "EPIC_100_noun_classes.csv",
    "EPIC_100_tail_nouns.csv",
    "EPIC_100_tail_verbs.csv",
    "EPIC_100_test_timestamps.csv",
    "EPIC_100_train.csv",
    "EPIC_100_unseen_participant_ids_test.csv",
    "EPIC_100_unseen_participant_ids_validation.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_verb_classes.csv",
    "EPIC_100_video_info.csv",
]


class Column:
    ACTION: str = "action"
    ACTION_ID: str = "action_id"
    END_TIME: str = "end_time"
    START_TIME: str = "start_time"
    SEQUENCE_LENGTH: str = "sequence_length"


def download_data(path: Path, force_download: bool = False) -> None:
    r"""Download the EPIC-KITCHENS-100 annotations.

    Args:
        path: The path where to store the downloaded data.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import download_data
    >>> download_data(Path("/path/to/data/epic_kitchen_100/"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    if not is_annotation_path_ready(path) or force_download:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_file = tmp_path.joinpath("epic-kitchens-100.zip.partial")
            logger.info(f"downloading EPIC-KITCHENS-100 annotations data in {zip_file}...")
            download_url_to_file(ANNOTATION_URL, zip_file.as_posix(), progress=True)

            logger.info(f"extracting {zip_file} in {tmp_path}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmp_path)

            logger.info(f"moving extracted files to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            for filename in ANNOTATION_FILENAMES:
                tmp_path.joinpath(f"epic-kitchens-100-annotations-master/{filename}").rename(
                    path.joinpath(filename)
                )

    logger.info(f"EPIC-KITCHENS-100 annotation data are available in {path}")


def is_annotation_path_ready(path: Path) -> bool:
    r"""Indicate if the given path contains the EPIC-KITCHENS-100
    annotation data.

    Args:
        path: The path to check.

    Returns:
        ``True`` if the path contains the EPIC-KITCHENS-100 data,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.epic_kitchen_100 import is_annotation_path_ready
    >>> is_annotation_path_ready(Path("/path/to/data/"))
    False

    ```
    """
    path = sanitize_path(path)
    return all(path.joinpath(filename).is_file() for filename in ANNOTATION_FILENAMES)


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.DEBUG)

    path = Path(os.environ["ARCTIX_DATA_PATH"]).joinpath("epic_kitchen_100")
    download_data(path)
