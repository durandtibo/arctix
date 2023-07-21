from unittest.mock import patch

from arctix.reducers import BasicReducer, NumpyReducer, TorchReducer, auto_reducer
from arctix.testing import numpy_available, torch_available

##################################
#     Tests for auto_reducer     #
##################################


@torch_available
def test_auto_reducer_torch() -> None:
    assert isinstance(auto_reducer(), TorchReducer)


@numpy_available
def test_auto_reducer_numpy() -> None:
    with patch("arctix.reducers.utils.is_torch_available", lambda *args, **kwargs: False):
        assert isinstance(auto_reducer(), NumpyReducer)


def test_auto_reducer_basic() -> None:
    with patch("arctix.reducers.utils.is_torch_available", lambda *args, **kwargs: False):
        with patch("arctix.reducers.utils.is_numpy_available", lambda *args, **kwargs: False):
            assert isinstance(auto_reducer(), BasicReducer)
