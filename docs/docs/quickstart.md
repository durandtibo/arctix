# Quickstart

:book: This page gives a quick overview of the main function of `arctix`: `summary`
This function can be used to compute a string representation of complex/nested objects.
The motivation of the library is explained [here](index.md#motivation).
You should read this page if you want to learn how to use this function.
This page does not explain the internal behavior of these functions.

**Prerequisites:** You’ll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).
It is highly recommended to know a bit of [NumPy](https://numpy.org/doc/stable/user/quickstart.html)
or [PyTorch](https://pytorch.org/tutorials/).

## Summary

`arctix` provides a function `summary` to compute a string representation of complex/nested objects.
It also works for simple objects like integer or string.

### First example

The following example shows how to use the `summary` function.
The object to summarize is a dictionary containing a PyTorch `Tensor` and a NumPy `ndarray`.

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
```

### Number of items

It is possible to control the number of items to show in a nested objects.
For example if there is a list of 1000 tensors, it is usually not necessary to show all the items.
The default string function shows all the items.

```pycon
>>> import torch
>>> print(str([torch.ones(2, 3) for i in range(1000)]))
[tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        ...  # a lot of lines are not shown
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]]), tensor([[1., 1., 1.],
        [1., 1., 1.]])]
```

The `summary` function returns the representation of the first items in the list.

```pycon
>>> import torch
>>> from arctix import summary
>>> print(summary([torch.ones(2, 3) for i in range(1000)]))
<class 'list'> (length=1,000)
  (0): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (1): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (2): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (3): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (4): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  ...
```

By efault, the maximum number of items to show is 5, but it can be changed easily by using the
context manager `summarizer_options`.
The following example shows how to return the first 3 items.

```pycon
>>> import torch
>>> from arctix import summarizer_options, summary
>>> with summarizer_options(max_items=3):
...     print(summary([torch.ones(2, 3) for i in range(1000)]))
<class 'list'> (length=1,000)
  (0): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (1): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (2): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  ...
```

`max_items` controls the number of items to show in lists, tuples, sets, dicts, etc.
It is possible to set `max_items=-1` to return all the items:

```pycon
>>> import torch
>>> from arctix import summarizer_options, summary
>>> with summarizer_options(max_items=-1):
...     print(summary([torch.ones(2, 3) for i in range(1000)]))
<class 'list'> (length=1,000)
  (0): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (1): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (2): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  ...  # a lot of lines are not shown
  (997): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (998): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
  (999): <class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu
```

### Maximum depth

As for the maximum number of items, it is possible to control the depth where the summarization is
applied in the complex/nested object.
A larger `max_depth` usually leads to more detail in the string representation.

- **Maximum depth 0**

```pycon
>>> import torch
>>> from arctix import summary
>>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=0))
[[0, 1, 2], {'key1': 'abc', 'key2': 'def'}]
```

- **Maximum depth 1**

```pycon
>>> import torch
>>> from arctix import summary
>>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
<class 'list'> (length=2)
  (0): [0, 1, 2]
  (1): {'key1': 'abc', 'key2': 'def'}
```

- **Maximum depth 2**

```pycon
>>> import torch
>>> from arctix import summary
>>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
  (0): <class 'list'> (length=3)
      (0): 0
      (1): 1
      (2): 2
  (1): <class 'dict'> (length=2)
      (key1): abc
      (key2): def
```

- **Maximum depth 3**

```pycon
>>> import torch
>>> from arctix import summary
>>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=3))
 (0): <class 'list'> (length=3)
      (0): <class 'int'> 0
      (1): <class 'int'> 1
      (2): <class 'int'> 2
  (1): <class 'dict'> (length=2)
      (key1): <class 'str'> abc
      (key2): <class 'str'> def
```
