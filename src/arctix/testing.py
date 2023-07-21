from __future__ import annotations

__all__ = ["numpy_available", "numpy_or_torch_available", "torch_available"]

from coola.utils.imports import is_numpy_available, is_torch_available
from pytest import mark

numpy_available = mark.skipif(not is_numpy_available(), reason="Requires NumPy")
torch_available = mark.skipif(not is_torch_available(), reason="Requires PyTorch")
numpy_or_torch_available = mark.skipif(
    not (is_numpy_available() or is_torch_available()), reason="Requires NumPy or PyTorch"
)
