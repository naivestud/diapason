"""A deterministic adapter used in tests and examples.

Real adapters wrap a model or an endpoint. For tests and examples we need
something that produces predictions without any model at all, so the dummy
adapter derives its answer from the reference annotation. With no corruption
it simply echoes the reference (a perfect score); turning up ``corruption``
deterministically drops words so the score degrades in a reproducible way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diapason.adapters.base import ModelAdapter
from diapason.registry import adapters

if TYPE_CHECKING:
    from diapason.datasets.manifest import Example


@adapters.register("dummy")
class DummyAdapter(ModelAdapter):
    """Echo the reference answer, optionally corrupting it deterministically."""

    name = "dummy"

    def __init__(self, corruption: float = 0.0, **options: Any) -> None:
        super().__init__(corruption=corruption, **options)
        if not 0.0 <= corruption <= 1.0:
            raise ValueError(f"corruption must be within [0, 1], got {corruption}")
        self.corruption = corruption

    def predict(self, example: Example) -> str:
        """Return the reference text, dropping words when corrupted."""
        text = example.reference_text or ""
        if self.corruption <= 0.0 or not text:
            return text
        words = text.split()
        step = max(1, round(1.0 / self.corruption))
        kept = [word for index, word in enumerate(words) if index % step != step - 1]
        return " ".join(kept)
