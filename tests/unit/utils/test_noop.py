from __future__ import annotations

from arctix.utils.noop import tqdm


def test_tqdm() -> None:
    x = [1, 2, 3, 4]
    assert tqdm(x) is x
