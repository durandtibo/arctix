from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

from arctix.transformer.dataframe import Cast

logger = logging.getLogger(__name__)


def check_transformer() -> None:
    logger.info("Checking arctix.transformer package...")

    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.String, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def main() -> None:
    check_transformer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
