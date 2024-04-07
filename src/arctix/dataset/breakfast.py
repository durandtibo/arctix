r"""Contain code to download and prepare the Breakfast data.

Information about the Breakfast dataset can be found in the following
paper:

The Language of Actions: Recovering the Syntax and Semantics of Goal-
Directed Human Activities. Kuehne, Arslan, and Serre. CVPR 2014.

Project page:

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

Data can be downloaded at

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/#Downloads

The documentation assumes the data are downloaded in the directory `/path/to/data/breakfast/`.
"""

from __future__ import annotations

__all__ = ["download_annotations"]

import logging
import tarfile
from pathlib import Path

from iden.utils.path import sanitize_path

from arctix.utils.download import download_drive_file

logger = logging.getLogger(__name__)

URLS = {
    "segmentation_coarse": "https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E",
    "segmentation_fine": "https://drive.google.com/open?id=1Alg_xjefEFOOpO_6_RnelWiNqbJlKhVF",
}
COOKING_ACTIVITIES = (
    "cereals",
    "coffee",
    "friedegg",
    "juice",
    "milk",
    "pancake",
    "salat",
    "sandwich",
    "scrambledegg",
    "tea",
)
NUM_COOKING_ACTIVITIES = {
    "cereals": 214,
    "coffee": 100,
    "friedegg": 198,
    "juice": 187,
    "milk": 224,
    "pancake": 173,
    "salat": 185,
    "sandwich": 197,
    "scrambledegg": 188,
    "tea": 223,
}


def download_annotations(path: Path, force_download: bool = False) -> None:
    r"""Download the Breakfast annotations.

    Args:
        path: The path where to store the downloaded data.
        force_download: If ``True``, the annotations are downloaded
            everytime this function is called. If ``False``,
            the annotations are downloaded only if the
            given path does not contain the annotation data.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from arctix.dataset.breakfast import download_annotations
    >>> download_annotations(Path("/path/to/data"))  # doctest: +SKIP

    ```
    """
    path = sanitize_path(path)
    logger.info(f"Downloading Breakfast dataset annotations in {path}...")
    for name, url in URLS.items():
        if not path.joinpath(name).is_dir() or force_download:
            tar_file = path.joinpath(f"{name}.tar.gz")
            download_drive_file(url, tar_file, quiet=False, fuzzy=True)
            tarfile.open(tar_file).extractall(path)  # noqa: S202
            tar_file.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)

    path = Path("~/Downloads/breakfast")
    download_annotations(path)
