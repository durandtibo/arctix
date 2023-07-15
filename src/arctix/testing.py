from __future__ import annotations

__all__ = ["torch_available"]

from pytest import mark

from arctix.utils.imports import is_torch_available

torch_available = mark.skipif(not is_torch_available(), reason="Requires PyTorch")
