# Home

<p align="center">
    <a href="https://github.com/durandtibo/arctix/actions">
        <img alt="CI" src="https://github.com/durandtibo/arctix/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://durandtibo.github.io/arctix/">
        <img alt="CI" src="https://github.com/durandtibo/arctix/workflows/Documentation/badge.svg?event=push&branch=main">
    </a>
    <a href="https://codecov.io/gh/durandtibo/arctix">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/arctix/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/arctix/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/61b8574ea18ecf106dce/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/arctix/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/61b8574ea18ecf106dce/test_coverage" />
    </a>
    <br/>
    <a href="https://pypi.org/project/arctix/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/arctix">
    </a>
    <a href="https://pypi.org/project/arctix/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/arctix.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/arctix">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <br/>
    <a href="https://pepy.tech/project/arctix">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/arctix">
    </a>
    <a href="https://pepy.tech/project/arctix">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/arctix/month">
    </a>
    <br/>
</p>

## Overview

`arctix` is a Python library that provides simple functions to check in a single line if two
complex/nested objects are equal or not.
`arctix` was initially designed to work
with [PyTorch `Tensor`s](https://pytorch.org/docs/stable/tensors.html)
and [NumPy `ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), but it
is possible to extend it
to [support other data structures](https://durandtibo.github.io/arctix/customization).

## Motivation

Let's imagine you have the following dictionaries that contain both a PyTorch `Tensor` and a
NumPy `ndarray`.
You want to compute a string representation of it.
By default, Python tries to show the values of all the tensor/array.
The `arctix` library was developed to easily compute structured string representation of nested
objects.
`arctix` provides a function `summary` to compute the string representation.

```pycon
>>> import numpy
>>> import torch
>>> from arctix import summary
>>> print(summary({"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}))
<class 'dict'> (length=2)
  (torch): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (numpy): <class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64
>>> print(
...     summary(
...         {
...             "torch": [torch.ones(2, 3), torch.zeros(6)],
...             "numpy": numpy.zeros((2, 3)),
...             "other": [42, 4.2, "abc"],
...         },
...         max_depth=3,
...     )
... )
<class 'dict'> (length=3)
  (torch): <class 'list'> (length=2)
      (0): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
      (1): <class 'torch.Tensor'> | shape=torch.Size([6]) | dtype=torch.float32 | device=cpu
  (numpy): <class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64
  (other): <class 'list'> (length=3)
      (0): <class 'int'> 42
      (1): <class 'float'> 4.2
      (2): <class 'str'> abc
```

Please check the [quickstart page](https://durandtibo.github.io/arctix/quickstart) to learn more on
how to use `arctix`.

## API stability

:warning: While `arctix` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `arctix` to a new version will possibly break any code that
was using the old version of `arctix`.

## License

`arctix` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.
