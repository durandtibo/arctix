from unittest.mock import Mock

from coola.utils import is_numpy_available
from pytest import mark, raises

from arctix import Summarizer, summary
from arctix.formatters.numpy_ import NDArrayFormatter
from arctix.testing import numpy_available

if is_numpy_available():
    import numpy
else:
    numpy = Mock()


@numpy_available
def test_summary_ndarray() -> None:
    assert summary(numpy.ones((2, 3))) == "<class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64"


#####################################
#     Tests for NDArrayFormatter     #
#####################################


@numpy_available
def test_ndarray_formatter_str() -> None:
    assert str(NDArrayFormatter()).startswith("NDArrayFormatter(")


@numpy_available
def test_ndarray_formatter_clone_show_data_10() -> None:
    formatter = NDArrayFormatter()
    formatter_cloned = formatter.clone()
    formatter.set_show_data(True)
    assert formatter is not formatter_cloned
    assert formatter.equal(NDArrayFormatter(show_data=True))
    assert formatter_cloned.equal(NDArrayFormatter())


@numpy_available
def test_ndarray_formatter_equal_true() -> None:
    assert NDArrayFormatter().equal(NDArrayFormatter())


@numpy_available
def test_ndarray_formatter_equal_false_different_show_data() -> None:
    assert not NDArrayFormatter().equal(NDArrayFormatter(show_data=True))


@numpy_available
def test_ndarray_formatter_equal_false_different_type() -> None:
    assert not NDArrayFormatter().equal(42)


@numpy_available
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
@mark.parametrize("dtype", (float, int, bool))
def test_ndarray_formatter_format(shape: tuple[int, ...], dtype: numpy.dtype) -> None:
    assert (
        NDArrayFormatter()
        .format(Summarizer(), numpy.ones(shape, dtype=dtype))
        .startswith("<class 'numpy.ndarray'>")
    )


@numpy_available
def test_ndarray_formatter_format_show_data_false() -> None:
    assert (
        NDArrayFormatter().format(Summarizer(), numpy.ones((2, 3)))
        == "<class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64"
    )


@numpy_available
def test_ndarray_formatter_format_show_data_true() -> None:
    print(NDArrayFormatter(show_data=True).format(Summarizer(), numpy.ones((2, 3))))
    assert (
        NDArrayFormatter(show_data=True).format(Summarizer(), numpy.ones((2, 3)))
        == "array([[1., 1., 1.],\n       [1., 1., 1.]])"
    )


@numpy_available
def test_ndarray_formatter_load_state_dict() -> None:
    formatter = NDArrayFormatter()
    formatter.load_state_dict({"show_data": True})
    assert formatter.equal(NDArrayFormatter(show_data=True))


@numpy_available
def test_ndarray_formatter_state_dict() -> None:
    assert NDArrayFormatter().state_dict() == {"show_data": False}


@numpy_available
def test_ndarray_formatter_get_show_data() -> None:
    assert not NDArrayFormatter().get_show_data()


@numpy_available
@mark.parametrize("show_data", (True, False))
def test_ndarray_formatter_set_show_data_int(show_data: bool) -> None:
    formatter = NDArrayFormatter()
    assert not formatter.get_show_data()
    formatter.set_show_data(show_data)
    assert formatter.get_show_data() == show_data


@numpy_available
def test_ndarray_formatter_set_show_data_incorrect_type() -> None:
    formatter = NDArrayFormatter()
    with raises(TypeError, match="Incorrect type for show_data. Expected bool value"):
        formatter.set_show_data(4.2)
