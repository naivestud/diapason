"""Model adapters that produce predictions for evaluation examples.

Importing this package imports every adapter submodule, which runs their
``@adapters.register`` decorators and makes the adapters available to suite
configs by name.
"""

from __future__ import annotations

from diapason.adapters.base import ModelAdapter
from diapason.adapters.dummy import DummyAdapter
from diapason.adapters.http import HttpAdapter

__all__ = ["DummyAdapter", "HttpAdapter", "ModelAdapter"]
