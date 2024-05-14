# arctix

<p align="center">
    <a href="https://github.com/durandtibo/arctix/actions">
        <img alt="CI" src="https://github.com/durandtibo/arctix/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/arctix/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/arctix/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/arctix/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/arctix/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/arctix/">
        <img alt="Documentation" src="https://github.com/durandtibo/arctix/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/arctix/">
        <img alt="Documentation" src="https://github.com/durandtibo/arctix/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
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
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
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

The `arctix` package consists of functionalities to prepare dataset of asynchronous time series.
It is design to make dataset preparation reusable and reproducible.
For each dataset, `arctix` provides 3 main functions:

- `fetch_data` to load the raw data are loaded in
  a [`polars.DataFrame`](https://docs.pola.rs/py-polars/html/reference/dataframe/index.html). When
  possible, it downloads automatically the data.
- `prepare_data` to prepare the data. It outputs the prepared data
  in [`polars.DataFrame`](https://docs.pola.rs/py-polars/html/reference/dataframe/index.html), and
  the metadata.
- `to_array` to convert the prepared data to a dictionary of numpy arrays.

For example, it is possible to use the following lines to download and prepare the MultiTHUMOS data.

```pycon

>>> from pathlib import Path
>>> from arctix.dataset.multithumos import fetch_data, prepare_data, to_array
>>> dataset_path = Path("/path/to/dataset/multithumos")
>>> data_raw = fetch_data(dataset_path)  # doctest: +SKIP
>>> data, metadata = prepare_data(data_raw)  # doctest: +SKIP
>>> arrays = to_array(data)  # doctest: +SKIP

```

- [Documentation](https://durandtibo.github.io/arctix/)
- [Installation](#installation)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

## Documentation

- [latest (stable)](https://durandtibo.github.io/arctix/): documentation from the latest stable
  release.
- [main (unstable)](https://durandtibo.github.io/arctix/main/): documentation associated to the
  main branch of the repo. This documentation may contain a lot of work-in-progress/outdated/missing
  parts.

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`arctix` can be installed from pip using the following command:

```shell
pip install arctix
```

To make the package as slim as possible, only the minimal packages required to use `arctix` are
installed.
To include all the packages, you can use the following command:

```shell
pip install arctix[all]
```

Please check the [get started page](https://durandtibo.github.io/arctix/get_started) to see how to
install only some specific packages or other alternatives to install the library.
The following is the corresponding `karbonn` versions and dependencies.

| `batcharray` | `batcharray`   | `coola`      | `iden`           | `numpy`       | `polars`        | `python`      |
|--------------|----------------|--------------|------------------|---------------|-----------------|---------------|
| `main`       | `>=0.0.2,<0.1` | `>=0.3,<1.0` | `">=0.0.3,<1.0"` | `>=1.22,<2.0` | `>=0.20.0,<1.0` | `>=3.9,<3.13` |
| `0.0.5`      | `>=0.0.2,<0.1` | `>=0.3,<1.0` | `">=0.0.3,<1.0"` | `>=1.22,<2.0` | `>=0.20.0,<1.0` | `>=3.9,<3.13` |
| `0.0.4`      | `>=0.0.2,<0.1` | `>=0.3,<1.0` | `">=0.0.3,<1.0"` | `>=1.22,<2.0` | `>=0.20.0,<1.0` | `>=3.9,<3.13` |
| `0.0.3`      | `>=0.0.2,<0.1` | `>=0.3,<1.0` | `">=0.0.3,<1.0"` | `>=1.22,<2.0` | `>=0.20.0,<1.0` | `>=3.9,<3.13` |

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## API stability

:warning: While `arctix` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `arctix` to a new version will possibly break any code that
was using the old version of `arctix`.

## License

`arctix` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.
