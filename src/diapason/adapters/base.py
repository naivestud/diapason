"""Base class shared by every model adapter.

An adapter turns an :class:`~diapason.datasets.manifest.Example` into a text
prediction. Concrete adapters wrap anything that can answer a prompt: a local
model, a hosted endpoint, or a deterministic stand-in used in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diapason.datasets.manifest import Example


class AdapterError(RuntimeError):
    """Raised when an adapter cannot produce a prediction for an example."""


class ModelAdapter(ABC):
    """Produces text predictions one example at a time.

    Subclasses set :attr:`name` and implement :meth:`predict`. The harness
    constructs an adapter with the ``adapter_options`` mapping from the suite
    config, so constructor keyword arguments are part of the public surface.
    """

    #: Unique name referenced by suite configs.
    name: str = "base"

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def predict(self, example: Example) -> str:
        """Return the model's text prediction for a single example."""
