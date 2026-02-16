"""A tiny name-to-factory registry for pluggable components.

Adapters and metrics register themselves under a short string name so suite
configs can refer to them symbolically. The registry is deliberately small:
register, look up, and list.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a component is missing or registered twice."""


class Registry(Generic[T]):
    """Maps string names to factory callables.

    Args:
        kind: A human readable label used in error messages, e.g. ``"metric"``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers ``factory`` under ``name``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise RegistryError(f"{self._kind} '{name}' is already registered")
            self._items[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        """Look up a factory by name, raising :class:`RegistryError` if absent."""
        if name not in self._items:
            known = ", ".join(self.names()) or "<none>"
            raise RegistryError(f"unknown {self._kind}: '{name}' (known: {known})")
        return self._items[name]

    def names(self) -> list[str]:
        """Return the registered names, sorted for stable output."""
        return sorted(self._items)


#: Registry of model adapters, keyed by adapter name.
adapters: Registry[Any] = Registry("adapter")

#: Registry of scoring metrics, keyed by metric name.
metrics: Registry[Any] = Registry("metric")
