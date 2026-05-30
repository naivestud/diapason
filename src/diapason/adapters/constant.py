"""Adapter that returns one fixed text for every example.

A constant prediction is the simplest possible baseline: it scores every
example identically, so any real model should beat it. The adapter is also
handy when sanity-checking a new metric or a new suite wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diapason.adapters.base import ModelAdapter
from diapason.registry import adapters

if TYPE_CHECKING:
    from diapason.datasets.manifest import Example


@adapters.register("constant")
class ConstantAdapter(ModelAdapter):
    """Return the same fixed text for every example.

    Args:
        text: The prediction returned for every example.
    """

    name = "constant"

    def __init__(self, text: str = "", **options: Any) -> None:
        super().__init__(text=text, **options)
        self.text = text

    def predict(self, example: Example) -> str:
        """Return the fixed text regardless of the example."""
        return self.text
